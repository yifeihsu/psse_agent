# RL plan for the minimal environment, 2026-11-22

**Status (2026-11-22, 15:35 UTC): both stages are built; the two smokes passed on d6b7cfc; the cell is
redeployed at 98e6518, and stage A's run and both probes are submitted** (`mrp-steps` 24090332 to 24090338,
`mrp-probe-dev` 24090400 to 24090402, `mrp-probe-train` 24090446 to 24090450). The user approved the build
("Yes, build both"), the deploy and the smokes ("Yes, deploy and smoke"), then the redeploy, stage A's run and
both probes ("Yes, all three"). The cell is `research/hpc/minimal_rl_20261122`; its README has the order of
work, what each smoke showed and what was submitted (its section 9) and how a fork row is picked since the
probe's smoke (its section 10; section 10 below). Stage B and every further cluster action need the user's
word. Sections 1 to 8 are the plan as written before the build.

It replaces the "demonstration fix" I proposed on 2026-11-21 (a look at the breakers written into the supervised
rows). The user objected that a decision rule taught through demonstrations is a rule-based mechanism by another
route, and that the objective is to improve performance through RL. The second supervised cell
(`research/hpc/minimal_sft_20261121`) therefore stays undeployed, and the `on_alarm` rebuild is not made.

Background: `docs/minimal_env_20261115.md` (the environment and every result so far),
`research/hpc/minimal_rl_20261116/README.md` (the two RL runs), `research/hpc/minimal_diag_20261120/README.md`
(where the unrepaired episodes stop and how far the repairing action is).

## 1. Where the results stand

Solved of the 247 development cases that a truth-built sequence repairs:

| Policy | Solved of 247 |
| --- | --- |
| Rule expert | 184 |
| Supervised model (the start of every RL run) | 198 |
| Plain RL run, one update | 202 (gained 5, lost 1, p = 0.22) |
| Focused RL run, three updates | 199, 200, 202 (201 strict; gained 10, lost 6, p = 0.45) |

Neither RL run is distinguishable from the supervised model on this suite. 49 solvable cases are still failed.

## 2. What the finished runs say about the RL procedure

All numbers are from the focused run's own reports (`mrl-focus`, 2026-11-20 to 11-21) unless a source is named.

| | Measured |
| --- | --- |
| Optimizer steps in the whole run | 3: one per iteration, learning rate 1e-5 |
| Optimizer steps of the supervised pass that built the start | 3,706, learning rate 1e-4 falling linearly to zero |
| Sum of the learning rates (a bound on how far one weight can move) | 3e-5 for the RL run, about 0.185 for the supervised pass: about 1 to 6,000 |
| KL from the start after the second and third updates | 0.0033 and 0.0024: the policy has barely left the start |
| Wall time | 28.6 hours for the three iterations, 9.5 hours an optimizer step |
| Update phase | 4.5 hours (1,186 rows) and 3.0 hours (1,247 rows) in the first and third iterations, 8.6 to 13.7 s a row |
| Rollout and evaluation | about 1.2 hours for 256 sampled episodes; about 1.3 hours for the 300 development cases |
| Row gradients in the third update (1,247 rows) | median norm 7.6e-5, 95th percentile 0.048, largest 27.5; six rows reached the cap of 1.0; the step's norm was 3.6 |
| Sampled steps whose every token had probability above 0.99 | 36% (first and third rollouts) |
| Failed cases with a repaired sample among their eight | 11, 12 and 8 of the 24 drawn in the three iterations |
| Distance to the repairing action, the 49 failed development cases (diagnostic of 2026-11-20) | 25 within 3 nats, 24 beyond, 16 beyond 6 |
| Where the run's ten gains were | 9 within 2.2 nats; 1 at 8.2 nats |
| Probability of the needed look in the breaker-never-corrected cases | about 0.0003: one sample in about 3,300 |

**Reading (mine).** Two separate things hold RL back, and neither is evidence that RL cannot learn the decision.

1. **Almost no optimization.** Three steps at 1e-5 cannot move a choice that sits 5 nats away. Half of the update's
   rows together can carry at most 0.05 of a gradient of norm 3.6 (623 rows at a norm of at most 7.6e-5), so
   most of each nine-hour step is spent on rows that teach nothing.
2. **The better action is not sampled.** A policy-gradient update learns an action only from episodes that took
   it. Its expected weight on an action of probability p is proportional to p(1 - p), so at p = 0.0003 nothing
   arrives however many steps are taken. The supervised start makes this worse: its look habit was taught with
   hidden truth, and it stops with about 0.9 probability on finishing.

Stage A below addresses the first, Stage B the second.

## 3. Stage A: many more optimizer steps for the same compute

The procedure of the focused run, unchanged, with a real step budget. One small code change (the evaluation
interval); everything else is a setting that the update already has.

| Setting | Focused run | Stage A | Why |
| --- | --- | --- | --- |
| `update.optimizer_steps` | 1 | 8, dealt by case: 4 cases a step | each row is still used once, so the update costs the same and takes eight steps |
| `update.fresh_optimizer_steps` | not set | 1 | the first step of a new AdamW moves every weight by the learning rate whatever the gradient's size (defect of 2026-11-08) |
| `update.learning_rate` | 1e-5 | 3e-5 | the project's rate for a pass that continues an adapter |
| Evaluation | every iteration | every third iteration (new setting, to write) | 1.3 hours saved in two iterations of three |
| Iterations | 3 | 10 | |

Unchanged: the start (the supervised model, 198), the draw (24 unrepaired and 8 repaired training cases an
iteration), 8 samples a case at temperature 1, the reward (a repair, less damage, a false claim and 0.01 an
action), the KL term to the start (0.04), the truth-built examples for cases without a repaired sample, the
carried optimizer, whole cases per step, no clip.

**Cost (my arithmetic from the measured rates).** An iteration: rollout 1.2 hours, update 3 to 4.5 hours,
evaluation 1.3 hours every third: about 5 to 6 GPU-hours. Ten iterations: about 55 GPU-hours for 73 optimizer
steps (1 in the first update, 8 in each of the nine after), against about 29 hours for 3 steps so far. The sum of
the learning rates is 2.2e-3 against 3e-5: 73 times the focused run's.

**What it can reach.** The cases whose repair the sampling finds: 25 of the 49 failed development cases are
within 3 nats (in training, 162 of the 500 scored). The cases beyond are reached only through the truth-built
examples, whose log-likelihood term does not vanish for an unlikely action; the one far case the focused run
gained (8.2 nats) is consistent with that.

**Risks and how they are read.**

- More movement can lose solved cases (the focused run lost 6, three of them at all three evaluations). The KL
  term, the 8 solved cases drawn each iteration and the paired gained-and-lost count at every evaluation are the
  controls. If the first evaluation (after 17 steps) shows more lost than gained, the learning rate is halved.
- Steps 2 to 8 of an update train rows sampled before step 1 moved the weights. At 3e-5 a step this drift should
  be small; the update reports it for every step (`on_policy_check`). A large drift means fewer steps a batch.
- The step count and the learning rate are untested here. Every earlier RL config in this repository sets 1e-5
  and 1 to 4 steps an update, and the runs I know of took four iterations or fewer, so no measurement says that
  73 steps at 3e-5 are safe or enough.

**Optional saving, to measure first.** Rows whose every sampled token had probability above 0.99 (36% of the
sampled steps) have a per-token gradient of at most 0.01 times the advantage. Leaving them out of the policy term
would cut the update by about a third. I would switch the step diagnostics on in the smoke, read how much of the
gradient those rows carry, and only then decide.

## 4. Stage B: let RL see the outcome of actions it does not sample

New code. The idea is policy improvement by one-step lookahead, a standard RL method: try each action at a state
the policy reaches, let the policy continue, and move toward the action that earned more.

**What happens in an iteration, beside Stage A's sampling.**

1. For each drawn training case, the current model's greedy episode is run (the behaviour the evaluation measures).
2. At every state of that episode, each of the other five tools is forced once. The model writes the arguments
   itself; the harness fixes only the tool's name. The model then continues greedily to the end of the episode.
3. Every branch is scored with the same episode reward as any episode, prefix included.
4. At a state where a forced branch returned more than the model's own action, the update gets one row: that
   branch's action on that state's prompt, as a log-likelihood target weighted by the difference of the returns.
   A state where the model's own action was best gives no row.

**Why this is not a rule.** Nothing in it says when to look. The forks are at every state and over every tool, not
at states I pick. The target is whichever action earned more reward when the model itself carried on; the truth
enters only through the reward, as it does now. A look that changes nothing loses to finishing by the 0.01 action
cost, so the same mechanism teaches the model not to look where a look does not pay.

**Why it is not a screen at deployment.** The branching exists only in training. An evaluated or deployed episode
is one greedy pass, one action at a time, as now.

**Why a log-likelihood target and not the policy gradient.** The policy gradient's weight on an action of
probability p is proportional to p(1 - p). The earlier series credited decisions that way from forked branches
(workplan of 2026-11-10) and the model learned none of the 15 prize declines. A log-likelihood target has a
gradient proportional to 1 - p, which is at full strength exactly where the model is confidently wrong.

**What the diagnostic says to expect (mine, not measured).** At the stops of patterns A (second breaker left) and
E (faulted line never fitted), the cost is the tool: once the tool is taken, the right object follows at 0.0 and
0.4 nats. Pattern B (breaker never corrected) needs one forced look at the opening state. So one forced step
should reach A, B and E: 34 of the 49. Pattern D (a healthy line fitted, 9 cases) is a choice of object at 1.9
nats, which a forced tool with greedy arguments does not change; it needs the arguments sampled. Out of reach
either way: pattern F (2 cases, not learnable from what is shown) and the misspelled breaker name `CB_2R4_2R5`.

**Cost (my arithmetic).** 32 cases by about 4.6 states by 5 tools is about 740 branches; at about 3.5 generated
steps a branch and 4.7 s a step, about 3.4 GPU-hours an iteration, which shards over workers like the rollout.
The fork rows are few (at most one a state, about 150 an iteration), so they add minutes to the update.

**Truth-built examples in this stage: off (my recommendation, the user's decision).** With them off, every row of
the update comes from the outcome of the model's own actions, so a gain is RL's.

**To build, locally, with tests and no cluster action.**

- Generation that starts from a forced tool name and lets the model write the arguments (the policy has no such
  path today).
- A fork runner in the minimal environment: replay the prefix (as `research/minimal_env/stopping_states.py`
  already does), force the step, hand over to the model, score.
- The fork rows, and a `fork` phase in `research/minimal_env/rl_train.py` between the rollout and the update,
  sharded and restartable like the others.

**A reading before any training: the fork probe.** One job with the supervised model, no update: fork its greedy
episodes on the 49 failed development cases and on 100 unrepaired training cases, and count the cases that one
forced step repairs. About 5 and 10 GPU-hours (arithmetic). It is a labelled upper bound on what one round of
Stage B can teach, and it is read before a training run is spent. The development forks are for this reading
only and never become training rows.

## 5. Order, and the decisions that are the user's

Recommended order:

1. Write Stage A's evaluation interval and prepare its cell (prepared, not deployed).
2. Build Stage B locally.
3. On the user's word: the Stage A run and the fork probe, which can run at the same time.
4. Read both, then decide Stage B's run: from the supervised model (a clean comparison with Stage A) or from
   Stage A's best adapter (the best number).

Decisions:

| | Question | My recommendation |
| --- | --- | --- |
| D1 | Run Stage A as soon as its cell is ready, or wait until Stage B is built and run once? | Run A first: it needs almost no code and answers whether step budget alone moves the near cases |
| D2 | Truth-built examples in Stage B: off or on? | Off, so the gains are attributable to outcomes |
| D3 | Stage A at 8 steps a batch and 3e-5, or a more cautious setting? | 8 and 3e-5, with the halving rule above |
| D4 | The action cost stays 0.01? | Yes for now; mean actions are reported beside every result |

## 6. What I will not do

- Write a look rule, or any other decision rule, into demonstrations, a controller or a verifier.
- Deploy `minimal_sft_20261121` or build the `on_alarm` rows.
- Pick the fork states or the forked tools by hand.
- Branch at evaluation or deployment.
- Pin a GPU type or a partition.

The rule I measured on 2026-11-21 (the estimate still fails a test in 296 of 300 states with a second breaker
left, and in 122 of 2,091 with nothing left) is kept as a labelled reference only. After a run I would tabulate
when the trained model looks against whether the estimate fails: if RL learns the decision, its looks should move
toward the failing states without anything having told it to.

## 7. How a result is read

- Solved of the 247, with the cases gained and lost against the supervised model's 198, by failure pattern (A to
  I) and by distance band.
- What the suite can confirm: with 4 cases lost, 13 gained give p = 0.049; with 6 lost, 17 gained give p = 0.035.
  So a result this suite can tell from the supervised model is about 207 to 209 of the 247 or more.
- Mean actions an episode (4.57 after the focused run), since the goal includes fewer tool calls.
- On the training side: of the unrepaired cases drawn, how many the model's greedy episode repairs afterwards.

## 8. Not checked (as planned; the cell's README has the list after the build)

- Every cost in sections 3 and 4 is arithmetic from measured rates, not a measurement of these settings.
- Whether eight steps on one batch at 3e-5 stay close enough to the sampled policy.
- Whether a forced tool name followed by the model's own arguments decodes cleanly under the chat template.
- How many failed cases one forced step repairs: that is the fork probe's question.
- Whether log-likelihood rows on fork states, with the KL term, keep the cases already solved.

## 9. What the build changed (2026-11-22)

- **The forks are made in the rollout phase**, by the processes that sample, not in a phase of their own. The
  batch stage and the run's manifest are unchanged and the model is loaded once.
- **Stage A is counted at 9 iterations, not 10**: with an evaluation every third iteration the tenth would not be
  evaluated. Nine iterations are 65 optimizer steps (1, then 8 in each of eight updates), 41 to 55 GPU-hours by the
  same arithmetic, with evaluations after 17, 41 and 65 steps.
- **A forced call that does not render** as the supervised pass taught it gives no row and is counted; it does
  not stop the run.
- **Sampled arguments are an option** (`forks.argument_samples`, 0 in both configs): branches whose forced call
  is sampled, which try other objects of one tool. Only the probe's smoke uses it.
- **Local tests** (stand-ins for the model, the real environment and scoring): a stand-in that corrects one of
  two breakers and claims the case is repaired by one forced step, and its better branch is taught; a stand-in
  that claims a two-breaker case at once is not repaired by one forced step, and the probe reports that.

## 10. What the smokes showed and changed (2026-11-22)

Both smokes completed on the cluster; the cell's README (sections 9 and 10) has the tables.

- **Stage A's smoke.** The first update took one step and the second four with the optimizer carried; the first
  evaluation was skipped and the second made; integrity ok. Within one batch the steps drift from the policy that
  sampled it: the mean absolute log-ratio on uncertain tokens was 0.011, 0.14, 0.18 and 0.30 over the four steps,
  and the KL to the supervised model reached 0.021 after five steps (the focused run: 0.0033 and 0.0024 in all).
  An update row took 12.2 to 12.8 s, so more steps a batch cost nothing.
- **The probe's smoke.** The forced tool name tokenizes as the model writes it in 176 of 176 branches; no forced
  call was invalid and no replay drifted. One forced step repairs 3 of the 4 unrepaired development cases it ran
  on. Four cases are not a rate, and in one of them a call the environment refused, which changes nothing but the
  history, is followed by a repair as well.
- **Step 4 of section 4 is changed.** A state's best branch is now compared with what going on as the greedy
  episode did can still reach (the greedy return, or a branch kept at a later state), and the excess is the row's
  weight. Against the greedy return alone, the smoke's records gave 15 rows of which 8 were `finish`: stopping at
  the first state was taught beside the look that repairs the case two states later, and where the greedy episode
  ends in a false claim, stopping before looking at anything got weight 0.52. Both outcomes of `finish` are now
  forked as whole calls at every state, so the honest declaration is taught where the false claim was made. With
  the new comparison the same records give 10 rows, 86% of their weight on branches that repair. Deployed at
  98e6518 before the probes were submitted.
