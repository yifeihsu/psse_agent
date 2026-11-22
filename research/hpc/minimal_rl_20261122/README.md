# The RL plan of 2026-11-22: stage A, stage B and the fork probe

**Status: DEPLOYED 2026-11-22 at d6b7cfc; the two smokes are submitted; no run and no probe is submitted.**
The user's word for both: "Yes, deploy and smoke". Section 9 has the cluster record. Stage A's run, the probe and
stage B each still need the user's word.

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
- `psse_env/minimal/forks.py`: the model's own greedy episode, then at every state every other tool forced once,
  the model continuing greedily, each branch scored with the episode reward. A replay is checked against a digest
  of what the model read; a branch that drifted, a forced call that is not a valid call of its tool, and a call
  the environment refused are marked and never taught.
- `research/minimal_env/rl_train.py`, `forks`: the forks are made in the rollout phase, by the same processes that
  sample (one model load, one file per case, sharded and restartable like the samples). At a state where a branch
  returned more than the greedy episode, the update gets one supervised row: the branch's call on that state's
  prompt, weighted by the difference of the returns.
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

- Nothing ran on the cluster. The forced call, the forks and the probe have met the model nowhere yet.
- Whether `<|tool_call>call:NAME{` tokenizes to the first tokens of a call's own rendering. The probe's smoke
  counts it.
- Whether eight steps on one batch at 3e-5 stay close to the sampled policy, and whether 65 steps are safe or
  enough. No earlier RL run of this project took more than a few steps.
- Every cost above.
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

Both were pending when submitted. Results are added here when the jobs end.
