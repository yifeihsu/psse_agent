# The minimal environment: what was built and the first numbers (2026-11-15)

Branch `claude/rl-project-restructure-6526b4`, on top of `claude/open-rules-20261112` (ce6c0e6). Code:
`psse_env/minimal/` and `research/minimal_env/`. The contract of the environment is in `psse_env/minimal/README.md`;
this note records why it exists, how each point of the review was handled, and what the first runs show. All runs
here are local CPU runs. Nothing was trained and nothing was sent to the cluster.

## 1. Why

The review of the open-rules work (2026-11-15) read the clone's result as two facts at once. On the 300 development
cases the adapter cloned on truth-guided rows solved 163, against 186 for S0 in the same rule-relaxed environment:
it gained on the 63 cases only truth-built sequences solve (6 to 20) and lost more on the 184 the rule expert solves
(180 to 143). It also made 54 meter corrections after the estimator already passed, all on healthy meters, all
accepted by the old verifier.

The review's conclusion: the trouble is not which trajectories are imitated but that rules still decide what the
model may try, what is kept and when the episode ends. Its recommendation: keep the numerical tools, hand the
diagnostic decisions to the model, and move the truth to data generation and scoring. Build that as one clean
pipeline beside the old one, not as more switches on it.

## 2. What each recommendation became

| Review | In the code | State |
| --- | --- | --- |
| Separate computation, decision and scoring | `kernel.py` and `versions.py`; `env.py`, `actions.py`, `observation.py`; `scoring.py`, `truth.py` | built |
| `route_status`, `supported_corrections` no longer permit actions | the kernel never calls the provider's `get_*_context`; a target only has to exist | built |
| The verifier returns numbers, not `ACCEPT` / `REJECT` | a correction returns the changes and the estimate before and after; the candidate oracle is not called | built |
| No next-step hints from the controller | no `valid_next_actions`; the process oracle is not instantiated | built |
| A direct finish, no hand-off string | `finish` with `resolved` or `unresolved`, legal at any step | built |
| Format, object, version and numerical checks stay | `actions.validate`, the environment's existence checks, tool failures | built |
| Six actions; a correction saves, executes, re-estimates and reports | `inspect`, `correct_topology`, `correct_parameters`, `correct_measurements`, `undo`, `finish` | built |
| A state that does not converge is returned, not refused | saved and reported as not converged; `undo` works from it | built, tested with a stubbed solve |
| Record model calls, WLS runs, fits and wall-clock | `CostMeter`, `env.cost()` | built |
| A short prompt that lists no rule | `observation.system_prompt`: the task, then what the columns mean | built |
| Current state, actual modifications, action history | the three parts of the user turn | built |
| The same observation constructor in training and at run time | rows and the model policy both call `observation.messages` and `observation.tools` | built, tested |
| No object beyond reach because a rule did not rank it | every bus, line, meter and breaker is in the tables; any breaker can be re-estimated by name | built |
| Re-execute the truth in the new environment | `truth.build_success`, run on the 300 development and 2,550 training cases | built, run |
| Direct sequences for simple cases, search only for coupled ones | breakers, lines, meters first; other orders only when that fails | built |
| States that need no further correction | every path ends with `finish`; a recovery path can stage its error after the last true correction | built |
| Recovery paths after a wrong action | `truth.build_recovery`: wrong correction, `undo`, the rest | built, with simulated errors |
| Keep the cases the expert solves | the selection takes every case | built |
| Do not hand the student the hidden fault list | the environment never receives the audit part; tested | built |
| Supervise the whole decision, no keep/undo emphasis | one row per step: inspect, correct, undo, finish | built |
| Reward: outcome, cost, damage; no workflow terms | `scoring.episode_reward` | built |
| Damage from what changed, not from which target was named | `scoring.physical_comparison` | built |
| `finish` grants nothing | declared, estimator passing and repaired are three separate fields | built |
| Oracle sequences never mixed into RL groups | `rollout.rollout_group` holds only sampled episodes | built |
| KL reference = the new supervised model | the update's reference is the run's starting adapter; an RL run started from the new adapter has it | by construction, no run yet |
| The expert under the new interface, by mechanical translation | `expert.py`, replayed on the 300 development cases | built, run |
| Train the new supervised model | `research/minimal_env/train_sft.py`, rows and selection | trained (the pass finished 2026-11-18) and evaluated once: 198 of 247 (the cell's README) |
| RL from that checkpoint | `research/minimal_env/rl_train.py`: rollout, update, evaluation (section 12) | one iteration run (2026-11-18): 202 of 247; the focused variant deployed 2026-11-19; its training pass: the supervised model leaves 502 of 2,093 solvable training cases; its smoke passed 2026-11-20; the focused run submitted 2026-11-20, to three iterations: 199 and 200 of 247 after one and two updates |
| Demonstrations that look before acting and answer a refusal (the stopping-state diagnostic, section 13) | `truth.py` look rules, refusal and breakers-last paths; `build_demonstrations`, `select_rows` (section 14) | built 2026-11-21: 30,960 training rows; the second pass's cell prepared, not deployed |

## 3. Checks

### The same numerical tools

| Check | Result |
| --- | --- |
| The estimate's numbers and its bus, line and meter tables equal the old provider's on the same state | equal on 4 generated cases (unit test) |
| The breaker table equals the old topology context's | equal: 73 breakers, 8 flip re-estimations (unit test) |
| The truth's corrections through the old environment (all rules lifted) and through the new one | the same final measurements, case and breaker statuses (unit test) |
| The rule expert's 300 recorded development episodes, translated and replayed | strict audit agrees with the recorded outcome on 300 of 300 (184 solved) |
| Truth-built sequences, old environment against new, 300 development cases | 247 against 247, the same count in each of the 14 classes, the same failure reasons (53, 21, 1) |

### The truth never reaches the model

- A case whose audit part was replaced by another case's gives the same prompts at every step (unit test).
- An execution part that carries a truth field is refused (unit test).
- Metadata outside the evidence profile is dropped before any tool runs (unit test).
- No observation holds a private key, and no prompt holds `ACCEPT`, `REJECT`, `supported_corrections`,
  `route_status` or `valid_next_actions` (unit test).

80 tests: `python -m pytest psse_env/minimal research/minimal_env -q` (about 3 minutes). The cell's 17 run apart:
`python -m pytest research/hpc/minimal_sft_20261115/test_cell.py -q`.

## 4. Reference numbers on the 300 development cases

These are the comparisons the review asks to keep. Every row has its number now; the rows recorded in the old
environment have no object-level audit.

| Policy | Environment | Solved (strict audit) | Solved (object level) | Actions per case |
| --- | --- | --- | --- | --- |
| Rule expert (guarded variant), recorded | old | 184 | | 12.97 |
| The same episodes, translated and replayed | minimal | 184 | 187 | 6.53 |
| Truth-built sequences, recorded | old, five rules lifted | 247 | | 11.12 |
| Truth-built sequences | minimal | 247 | 247 | 4.24 |
| Supervised model on the new rows, one greedy pass (2026-11-18) | minimal | 198 | 198 | 4.45 |
| RL from that model, one iteration (2026-11-18) | minimal | 202 | 202 | 4.36 |

For orientation, in the old environment with the rules lifted: S0 186, the clone 163.

**The basis from 2026-11-18 on is the 247 cases a truth-built sequence repairs**, not the 300 (the user's rule):
supervised model 198 of 247 (80%), rule expert 184 of 247 (74%), 49 left.

**The supervised model (2026-11-18): 198.** It solves 161 of the 184 cases the rule expert solves, 37 of the 63
only the truth-built sequence repairs, and none of the 53 neither repairs: 37 gained and 23 lost against the
expert, in 4.45 actions a case against the expert's 6.53. The lead of 14 is from one greedy pass (two-sided sign
test on the 60 differing cases, p = 0.09, my arithmetic). It is behind both references in the three two-breaker
classes (15 of 35; the expert 21, the truth-built sequences 35). Of the 102 cases it does not solve, 53 are
the ones neither reference repairs and 49 are ones the truth-built sequence repairs. It declared 72 unsolved
cases resolved, 52 of them on a passing estimate. The classes, the endings and the refused calls are in
`research/hpc/minimal_sft_20261115/README.md`.

**One RL iteration from it (2026-11-18): 202 of 247.** 5 cases gained and 1 lost against the supervised model,
both passes greedy in bf16 on RTX PRO 6000 cards. Two of the gains are two-breaker cases where the supervised
model sent one refused topology correction fifteen times; the updated model does that in another case instead.
Five against one is a split two equally likely directions give about one time in five (two-sided sign test,
p = 0.22): one iteration does not show that the update helps. `research/hpc/minimal_rl_20261116/README.md` has the
rollout, the update and each case that moved.

Reading the rows:

- **The interface change alone changes no outcome.** The rule expert's decisions give 184 in either interface, in
  half the actions.
- **The two audits differ on 3 replayed expert episodes.** In all three the final state is within tolerance of the
  clean one, and the strict audit fails the episode because a correction in effect named a healthy line (2) or
  because an out-of-service line ended attached to another bus section (1). This is the difference the review
  describes between judging the target named and judging what changed.
- **247 is the coverage of these truth-built sequences, not a ceiling.** The other 53 end outside tolerance with the
  true corrections applied (53 line estimates outside tolerance, 21 of them with a line fit that failed).
- **The estimator passing is not the answer.** At the end of the 300 truth-built sequences the estimator passes on
  274. It passes on 40 of the 53 cases that are not repaired, and still alarms on 13 of the 247 that are. The
  replayed rule expert ends 245 episodes on a passing estimate, 69 of them not repaired.

## 5. Cost

Per development case, mean:

| | Actions | WLS solves | Substation solves | Line fits | Meter repairs |
| --- | --- | --- | --- | --- | --- |
| Truth-built, minimal | 4.24 | 2.96 | 10.05 | 1.42 | 0.33 |
| Rule expert replayed, minimal | 6.53 | 4.19 | 6.51 | 3.22 | 0.40 |

The old interface's 12.97 and 11.12 actions are not comparable with the 6.53 and 4.24: one correction there is four
actions. The solves are the comparable quantity. The old records count only the solves the policy asked for (1,040
`run_wls` calls over the 300 expert episodes), not the ones inside its contexts and comparisons, so an old-against-new
table of solves cannot be made from them; the replay's own count is 1,257 WLS solves and 966 line fits.

The truth-built sequences spend most of their substation solves on the inspection at the start of every case (one
estimate and 8 flip re-estimations). That inspection is a choice, see section 8.

## 6. Prompt size

Every digit is a token, so the tables are expensive: one version's bus, line and meter tables are about 4,000
tokens, a breaker table about 2,800, the system prompt 504, the tool list 729.

| Tables shown | Median | 95th percentile | Maximum |
| --- | --- | --- | --- |
| The two most recent versions (default) | 9,282 | 12,049 | 14,276 |
| Every version | 11,451 | 22,403 | 26,596 |

169 rows of 28 development cases (two per class), system and user turns, counted with the local
`unsloth/Gemma-4-26B-A4B-it` tokenizer file: an estimate for the 12B model, not its row gate. For comparison, the
complete-evidence selection of the old environment was estimated at a median of 9,095 and a maximum of 20,662.

## 7. Training data

Built on the 2,550 training cases (`output/compound_rl_20261106/train_ieee14/suite.json`, the suite of the earlier
cloning passes), 21 minutes on 24 local workers.

**Success paths.**

| | Old environment, five rules lifted | Minimal environment |
| --- | --- | --- |
| Cases repaired (strict audit) | 2,091 | 2,091, the same count in each of the 14 classes |
| Orders tried: 1 / 2 / 3 / 6 | 2,052 / 339 / 13 / 146 | 2,052 / 339 / 13 / 146 |
| Actions per case | 11.1 | 4.26 |

The object-level audit calls 2,093 repaired (one more in `meter_rx2` and one in `rx2_distant`). At the end of the
2,550 sequences the estimator passes on 2,272: on 305 of the 459 that are not repaired, and not on 126 of the
repaired ones.

**Recovery paths.** One per repaired case where an error could be staged: 1,855, every one repaired again after the
`undo`. The staged error is a meter correction in 722, a breaker in 681, a line fit in 452. It comes before the first
true correction in 602, after one in 680, after two in 472, after three in 101. In many of them the wrong correction
lowers the chi-square, which is what makes them worth showing: for example a healthy voltage meter moved by 39
standard deviations takes the statistic from 2,861 to 2,008, and the demonstration undoes it.

**Rows.** 16,077 rows, each checked against the schema half of the trainer's row gate when written.

**Selection** (`select_rows`, defaults: every case, a 6% share of each class held out whole):

| | Cases | Rows |
| --- | --- | --- |
| Training view | 2,398 | 14,823 |
| Validation view | 152 | 256 (sampled from 944, balanced over step kinds) |
| Left out | | 305 `finish unresolved` rows on a passing estimate |

Training rows by action: `correct_parameters` 3,219, `correct_topology` 1,989, `correct_measurements` 1,213,
`inspect` 2,814, `undo` 1,738, `finish` 3,850. By kind: success path 9,524 (corrections 4,722, inspections 2,690,
finishes 2,112), recovery path 5,299 (undo 1,738, corrections 1,699, finishes 1,738, inspections 124).

Of the 2,550 cases, 2,091 give all their rows and 459 are beyond the correction tools: those give their true-target
corrections and, in 152 of them, a `finish unresolved` on a final estimate that still alarms.

**Prompt length of the training view** (a 5% sample, 780 rows, the same tokenizer file): median 9,210 tokens, 95th
percentile 11,860, maximum 14,412, about 134 million prompt tokens in all. The complete-evidence selection was
estimated at 96.6 million for 11,061 rows, so a pass over this view is about 1.4 times that one's token volume. This
is my estimate from a sibling tokenizer; the row gate on the cluster gives the real figure.

Checksums: training view `582349e2...`, validation view `e55689a6...` (`selection_report.json`).

## 8. Choices made here that are yours to change

1. **Tables of two versions in the prompt.** Older versions keep their summary numbers and are one `inspect` away.
   Showing every version is one argument (`table_versions=None`) and costs up to 26,600 tokens instead of 14,300.
2. **An inspection of the breakers at the start of every demonstration.** Without it, whether to look at the breakers
   would be decided by the truth (inspect only when a breaker is wrong), which a student cannot imitate from the
   evidence. With it, every demonstration starts with the same action and the breaker table is in view for the first
   decision. It costs one action and 9 substation solves per case, and about 2,800 prompt tokens while that version
   is shown. `--no-opening-inspection` builds the other kind.
3. **Simulated errors in the recovery paths.** The wrong correction is the healthy meter with the largest residual,
   the healthy line with the largest multiplier or the healthy breaker with the largest score. This is my stand-in
   for a student's mistake until a trained model's own failed episodes exist.
4. **`finish unresolved` rows.** Kept only where the final estimate still alarms; dropped where it passes, because
   the prompt cannot tell that state from a repaired one.
5. **The reward weights** in `scoring.RewardWeights`: success 1.0, repaired but not declared 0.5, false `resolved`
   0.5, damage 0.25 per object up to 0.5, 0.01 per action.
6. **The claim of the replayed expert.** The old hand-off closes every episode with the same request, so the replay
   declares `resolved` when the final estimate passes. The expert's number above is `solved`, which does not depend
   on this.
7. **Tool limits left as they are.** A breaker flip that would isolate equipment (not a line terminal, a split or a
   merge) is refused by the breaker tool as before; the line fit and the meter repair are the old ones. Nothing in
   this work changes what the tools can do, so the 53 unrepaired cases stay unrepaired.

## 9. Not done

- **The supervised model is trained and evaluated once** (2026-11-18, section 4). Not done with it: a second
  greedy pass on another card type, and rows that follow a refused call (two episodes repeated one refused call
  15 times).
- **One RL iteration** (section 12): its cluster cell (`research/hpc/minimal_rl_20261116/`) ran the smoke on
  2026-11-17 and one iteration of the run on 2026-11-18, 202 of 247 against the supervised model's 198 (section
  4). The focused run (deployed 2026-11-19 at 2d4919d; its training pass leaves 502 of the 2,093 solvable
  training cases unrepaired; its smoke passed and the run was submitted on 2026-11-20, to three iterations):
  199 of 247 after one update, 200 after two (gained 7, lost 5 against the supervised model); the third
  iteration is running.
- **IEEE 118** is not handled: no breaker model there (the line-status form of `correct_topology` exists and is
  untested on that system), and a version's meter table is about 1,100 rows.
- **The waveform families** (HIF, unbalance, harmonics) are outside this environment.
- The full repository suite was not rerun: no existing module was changed, only two README files.

## 10. Where things are

| What | Where |
| --- | --- |
| Environment and tests | `psse_env/minimal/` |
| Builders, baseline, selection, trainer entry point, evaluation, RL driver | `research/minimal_env/` |
| Truth reference, 300 development cases | `output/minimal_env_20261115/truth_reference_dev300/` |
| Rule expert replay, 300 development cases | `output/minimal_env_20261115/expert_dev300/` |
| Demonstrations and rows, 2,550 training cases | `output/minimal_env_20261115/demonstrations_train_ieee14/` |
| Training and validation views | `output/minimal_env_20261115/selection_ieee14/` |
| The four runs' summaries, committed | `docs/minimal_env_20261115/` |

`output/` is ignored by git and sits in this session's worktree
(`.claude/worktrees/dreamy-bassi-27849a/output/minimal_env_20261115/`, 555 MB). The runs were made before the
commit, so their manifests name the parent commit (ce6c0e6); the code that made them is this commit's. The development suite and the rule expert's
recorded episodes were read from the worktree `nostalgic-kalam-be5f7a`
(`output/compound_rl_20261107/ieee14_dev300.json`, `output/compound_rl_20261106/reference_ieee14/teacher/episodes.jsonl`);
the training suite from the main checkout (`output/compound_rl_20261106/train_ieee14/suite.json`).

## 11. The cell for the supervised pass (prepared 2026-11-15)

This section is the cell as it was prepared. It has since been deployed, smoked and submitted (2026-11-16); what
ran and what it showed is in the cell's README.

`research/hpc/minimal_sft_20261115/` (its README has the order of work). Three jobs: the tokenizer row gate on a CPU
node, the pass on the 14,823 training rows (3,706 optimizer steps), and the evaluation of its adapter on the 300
development cases in the minimal environment, greedy and in bf16.

**The start is the base model** (the user's choice of 2026-11-15): a new LoRA, nothing inherited from the rule
expert's demonstrations. The cell can also start from R1, the adapter every earlier cloning pass started from, and
keeps the two apart. Two trainer settings follow the start: no initial adapter, and a learning rate of 1e-4 instead
of 3e-5. 1e-4 is what BC0, the project's one earlier pass from the base model, trained at; 3e-5 is the rate of every
pass that continued an adapter. Everything else is the cloning passes' setting, one for one.

The evaluation's summary splits the 300 cases as the review reads a result (`dev300_reference.json`, built from the
two runs of section 4): the 184 the rule expert's recorded episodes solve, the 63 only the truth-built sequence
repairs, and the 53 neither repairs.

My estimate for the pass is 41 to 62 GPU-hours (40 to 60 s per step for prompts of about 9,900 tokens with their
tools), 7 links of 12 hours as submitted; the evaluation 2 to 4 hours. Nothing in the cell has run with the model or
its tokenizer. The smoke is the first time the rows meet the chat template.

## 12. The RL driver (built 2026-11-16, no run)

`research/minimal_env/rl_train.py`, with an example config beside it (`rl_config_example.json`).

| Phase | What it does | What it leaves in `iteration_NNN/` |
| --- | --- | --- |
| `rollout` | K sampled episodes of every drawn training case, the current adapter choosing every action | `trajectories/`, one file per case; `rollout_summary.json` |
| `update` | one policy-gradient update on the sampled tokens; an episode's advantage is its reward less the mean reward of its case's samples | `adapter/`, `advantages.json`, `update_report.json` |
| `evaluate` | the updated adapter, greedy, on the development cases, written as the supervised pass's evaluation is | `evaluation.json`, with the cases gained and lost against the run's baseline |

A run can start with a `baseline` phase: the starting adapter evaluated the same way. `manifest.json` holds the
phase the run is at and moves on only when a phase completes, so every phase resumes after a preemption. The
rollout and the evaluations split over worker processes (`--shard i/n`); the call without a shard finishes what is
left and closes the phase.

**What is reused and what is new.** The trajectory store, the group advantages, the row builder, the update with
its checkpoint and carried optimizer, and the integrity checks are the RL series' (`psse_env/rl`), called as they
are, so the repairs of 2026-11-09 to 11 are in. The episode, the reward, the prompt and the tools are this
environment's. There is no lifecycle label, no decision credit and no supervised target: a correction is kept or
undone by the model's own later action and credited by the episode's outcome. Truth-built sequences are never part
of a group. No old module was changed.

**Settings of the example config, and why.**

1. **The advantage is not divided by the group's spread.** Samples of a case that all repair it still differ by an
   action or two. Dividing by the spread would turn that 0.01 of cost into an advantage as large as the one between
   a repair and a failure.
2. **One optimizer step per update, the optimizer carried from update to update, no clip.** Two steps of a new
   AdamW move every weight the way the first half of the batch points (measured on the first arm, 2026-11-09). With
   one step there is nothing for a clip to bound, and its stored baseline differs between card types.
3. **Every sampled token weighs the same (`dapo`).** It is the one normalization under which what a case's samples
   share cancels exactly.
4. **A KL penalty to the run's starting adapter**, the supervised model, never to the adapter of the iteration
   before. Cases whose samples all scored the same add rows for this term alone.
5. **Only what was sampled is trained.** A case counts only with all its samples, and a step only if its stored
   observation renders again to the prompt it was sampled from (same token digest, never cut). A step that fails
   this stops the run; it does not shrink the batch quietly.
6. **bf16 for the rollouts, the update and the evaluations**, and sampling at temperature 1 from the model's whole
   distribution (no nucleus or top-k cut, which the update could not account for).

Carried over from the first arm without a new decision: 64 cases by 4 samples per iteration, drawn by class with
that arm's counts, a learning rate of 1e-5 and a KL weight of 0.04. These are yours to set when a cell is prepared.

**Checked.** 14 tests (`research/minimal_env/test_rl_train.py`) on generated cases. The environment, the scoring,
the store, the row builder, the update and the integrity checks are the real ones; the policy, the tokenizer and
the trained model are stand-ins. They cover one whole iteration from the baseline to the next rollout with sharded
calls; an update cut after its first step and resumed to the same weights; samples that all score the same (the
adapter is carried over and not evaluated again); a prompt that does not render back (the run stops and stays
stopped); the KL reference; the seeds and the adapter of the second iteration; and the real prompt renderer, which
hands the template this environment's messages and six tools from a stored step. 16 faults put into the driver by
hand each fail a test. The example config's class counts fit the real training suite (2,550 cases, 14 classes):
three iterations draw 192 different cases, and the development cases share none with it.

**Not checked: everything that needs the model.** A first smoke has to show:

- every sampled step carries its token probabilities, and the share of steps whose tokens are all above 0.99 (a
  policy that peaked samples the same episode K times and gives the update nothing);
- every stored step renders back to its sampled prompt under the model's own tokenizer (no digest mismatch);
- the gap between the sampler's log-probabilities and the update's first pass (the on-policy check);
- the update's memory on rows of 15,000 tokens with the reference adapter loaded (the first arm's update took
  about 24 GB plus 1 MB per prompt token, so about 40 GB is my expectation);
- how many cases have both a repaired and an unrepaired sample: that contrast is what an update can learn a repair
  from, and `rollout_summary.json` counts it.

**Cost, my estimate.** It rests on the supervised cell's smoke rates; the update's rate in bf16 is not measured.
Per iteration of 64 cases by 4 samples: the rollout 2 to 4 hours with one process (256 episodes of 4 to 7
actions, half that with two processes on a card); the update 4 to 7 hours (one row per sampled step, about 10 s a
row plus the reference's forward pass); the evaluation 1.5 to 3 hours. Together roughly 8 to 14 GPU-hours.

**After the smoke (2026-11-17).** Measured on an H200: 4.7 s a sampled step, 9.2 s an update row, 4.9 s an
evaluated step, two processes together 63.4 GB, the update 36.5 GB. From those rates an iteration of 64 by 4 is 3
to 6.5 GPU-hours on an H200 (my arithmetic). Every check of the update held. Two things for the run's first
iteration, from six and eight cases at a checkpoint 44% through the supervised pass: five of six cases sampled
the same episode twice and none had a repaired and an unrepaired sample; and an update with a gradient norm of
0.08 changed three of eight greedy episodes, which I read as the sign step a new AdamW takes. The cell's README
has both in full.

**The cluster cell** is `research/hpc/minimal_rl_20261116/` (deployed 2026-11-16 at e82026b; the smoke passed 2026-11-17
in 22 minutes; the run's first iteration finished 2026-11-18, 202 of 247): one stage that runs the phase the manifest is at, two processes per large card for the rollout and the evaluations, a smoke of
one small iteration from a checkpoint of the running supervised pass, and a chain for the run. The run starts from
the supervised pass's adapter and takes that cell's evaluation of it as its baseline (`baseline_from`), so the
supervised model has one number. Its README has the order of work and what the smoke has to show.

**The focused run (built 2026-11-18, deployed 2026-11-19 at 2d4919d; its training pass ran 2026-11-19: the
supervised model leaves 502 of the 2,093 solvable training cases unrepaired, 24.0% against 19.8% of the
development cases; the focused smoke passed on 2026-11-20: 6 of 6 replayed sequences repaired their cases, 25 of
25 example actions rendered, the update's integrity held; the focused run submitted on 2026-11-20 at 06:24 UTC,
to three iterations; its first iteration 199 of the 247, gained 5 and lost 4, a repaired sample in 11 of its 24
failed cases; its second 200, gained 7 and lost 5 against the supervised model, a repaired sample in 12 of 24).** After the supervised model's evaluation the user set the
basis to the 247 cases a truth-built sequence repairs and asked for a run aimed at the solvable cases the model
fails. `focus` in the driver's config does three things: it draws an iteration from the 2,093 training cases
with a truth-built sequence, 24 the supervised model leaves unrepaired and 8 it repairs (its greedy pass over
them decides); it takes eight samples a case; and where none of a case's samples is repaired it replays the
case's truth-built sequence and trains its actions as supervised rows, weighted by the sequence's reward above
the samples' mean (the user's "Yes, include them"). That last part is imitation of the truth-built sequence on
the cases the model fails, not on-policy data, and the driver keeps it apart: no sampling record, no advantage,
no part in the KL term or the on-policy check. 25 driver tests and 17 hand-made faults, each caught; nothing of
it has met the model. The cell's README has the jobs, the order of work and what its smoke has to show.

**Not built.** Recovery rows from the model's own failed episodes, which the rollouts will hold: truth-built
continuations from the state a failed episode reached. The focused run's examples start from the case's first
state.

## 13. The stopping-state diagnostic (built, part 1 and part 2 run 2026-11-20)

Asked after the training pass whether the focused RL can close the gap or the method needs work first; the user
said "Yes, build it".  `research/minimal_env/stopping_states.py` replays the supervised model's stored greedy
episodes (all 2,393 reproduce), labels every state with what the truth says is still wrong, and from the state
where each unrepaired episode stops, and from its first wrong turn, searches the truth-built way on;
`research/minimal_env/evidence_probe.py` asks whether the evidence there says the case is unrepaired and which
object is wrong; `research/minimal_env/score_actions.py` (cell `research/hpc/minimal_diag_20261120/`, deployed at
96dbf99 and its job `mdg-score` 23861813 submitted 2026-11-20 18:24Z on the user's "Deploy and submit"; it
completed in 2 h 15 min, all 785 states scored) reads the model's own probability of the step on.  The cell's
README has the tables.

- Of the 49 unrepaired development cases, 45 stop with a true error never corrected (29 of the 60 errors left are
  breakers), 3 with every error corrected and one fit outside tolerance, 1 with damage alone; every one has a
  truth-built way on (24 straight from the stop, 25 after an undo).  Training: 465, 23 and 14 of 502; 2 without.
- The evidence says the stop is unrepaired in 43 of 49 and places the object left first in 37 of 48, but for a
  breaker only on the current version's breaker table, one `inspect` away: with only what was shown, 2 of 23.
- After its first breaker correction in a two-breaker case the model inspects again in 140 of 289 training
  episodes (132 repaired) and not in 149 (5 repaired).  The truth-built sequences inspect there in all 300
  two-breaker cases and in none of the 891 others: that inspection is a decision the truth makes for the student.
- The failure patterns (cell README, "The failure patterns"), development / training of 49 / 502: a two-breaker
  case with the second breaker never corrected 17 / 144; a breaker never corrected and none corrected 7 / 90
  (mostly the one-breaker mixed classes; 2 / 13 of them the model's misspelling `CB_2R4_R5` of `CB_2R4_2R5`); a
  healthy breaker corrected instead 2 / 14; a healthy line fitted instead of the faulted one 9 / 56; a faulted line
  never fitted 10 / 159 (66 of the training ones after fitting the other faulted line); a line fit outside
  tolerance 2 / 17; a meter left 1 / 8; a healthy correction kept 1 / 14.  The focused run's rollouts reach a
  repair in 9 of 18 groups of the first, 8 of 15 of the faulted-line-never-fitted ones, 3 of 6 of the wrong-line
  ones and 2 of 7 of the breaker-never-corrected ones; the fit-outside-tolerance cases read as repaired on the
  evidence (4 of 17 flagged) and are a limit of the fitting tool, not of the policy.
- The scores (cell README, "Part 2"), at T = 1.  At the stop the model gives its own `finish` a median 0.89 in
  training (development 0.90) and the step on a median 0.00089 (0.00031); one of 8 samples would step on at about
  61 of the 502 training stops (7.2 of 49), and the median lead of the own action is 7.0 nats (8.2).  At the first
  wrong turn, where it comes earlier (209 / 25 states), the step on is closer: median 0.031 (0.075), one of 8 at 75
  of 209 (11 of 25), lead 3.2 nats (2.4).
- By pattern (training): the second breaker left and the faulted line never fitted cost the tool choice, not the
  object (5.3 and 4.2 nats against 0.0 and 0.4): one of 8 samples steps on at 29 of 144 and 24 of 159 stops, lead
  5.4 and 5.3 nats.  The breaker never corrected (none corrected) is out of reach: the `inspect` of the breakers at a
  median 0.0003, lead 8.2 nats, about 3 of 90 stops.  The wrong line is out of reach at the stop (an `undo` at 6e-9)
  but near at its first wrong turn: the faulted line at a median 0.10, one of 8 at 30 of 56, lead 1.9 nats, the cost
  the object.  The fit outside tolerance: lead 25.7 nats at the stop.  The misspelled `CB_2R4_2R5`: in all 17
  scored calls the `2` after `R4_` costs 6.0 to 11.3 nats and the rest of the name at most 0.2 (one at 1.8).
- Against the focused run's second evaluation (200 of 247 after two updates; cell README, "Against the focused
  run's second evaluation"): by each development case's smallest lead (stop or first wrong turn), 5 of the 7
  within 1 nat are solved, 2 of the 18 within 1 to 3 nats, none of the 24 beyond 3 (16 beyond 6, among them 6 of
  the 7 breaker-never-corrected cases).  The larger probability of a step on tells the solved cases from the
  others at an area under the curve of 0.92.

## 14. The second demonstration build (prepared 2026-11-21)

Asked after the stopping-state diagnostic and the focused run's second evaluation ("Yes, prepare it"): change the
demonstrations where the supervised model's failures start, so that looking at the breakers is never a decision
the truth makes, and a refused call is answered.  What the first demonstrations taught, and the change:

| | First build (2026-11-15) | Second build |
| --- | --- | --- |
| Look at the breakers after a breaker correction | only when another breaker correction follows (300 of 300 first-of-two, 0 of 1,300 others) | always (`--looks every_version`: before every action on a version not looked at, `finish` included) |
| A refused call in the history | only the truth's own refused corrections, retried | refusal paths: the most plausible healthy line fit, refused, on the version after the first breaker correction (or at the start), then the look and the rest |
| A breaker still wrong after the lines and meters | never shown (breakers are corrected first) | breakers-last paths: lines and meters first, the look that shows the breaker, the breaker, the refit of what was corrected on the wrong topology |
| `CB_2R4_2R5` | 36 training targets; the model wrote `CB_2R4_R5` in 245 of 263 calls | its rows written three times (`select_rows --repeat-breaker CB_2R4_2R5=3`) |

Code: `psse_env/minimal/truth.py` (`looks`, `build_refusal`, `build_breakers_last`; the default `needed` reproduces
the first build), `research/minimal_env/build_demonstrations.py` (`--looks`, `--refusals`, `--breakers-last`),
`research/minimal_env/select_rows.py` (`--skip-demonstrations`, `--repeat-breaker`); committed e95db77.

**The two look rules on a sample** (the first 20 cases of every class, 280 cases, against the first build's rows of
the same cases; tokens at 1.46 characters each, the first build's ratio):

| | First build | `after_breakers` | `every_version` |
| --- | --- | --- | --- |
| Look after an accepted breaker correction (success paths) | 60 of 220 (the first of two breakers only) | 220 of 220 | 220 of 220 |
| Success path, mean actions | 4.14 | 5.33 | 5.71 |
| Paths: success / recovery / refusal / breakers-last | 280 / 212 / - / - | 280 / 212 / 110 / 49 | 280 / 212 / 110 / 49 |
| Rows | 1,746 | 3,103 (x1.78) | 3,392 (x1.94) |
| Prompt tokens, median / max / all | 9,264 / 14,948 / 16.5M | 11,546 / 15,892 / 34.2M (x2.07) | 12,279 / 16,337 / 40.5M (x2.45) |
| Looks skipped for the action budget | - | 0 | 0 |

In the sample the refusal paths come from the two refusals the supervised model met (`parameter_correction_failure`
76, the scan-size error after a bus split 34), 72 after the first breaker correction and 38 at the start, in 47 of
the 60 two-breaker cases; the breakers-last paths come from the classes of pattern B (rx_cb_distant 16, rx_cb_meter
17, rx_cb_shared_bus 16 of 20 each), 17 of the 49 with a refit.  Every path kept repairs its case by both audits.
`every_version` was built: after a line fit the look comes before the student can fit another line on an estimate
that still alarms because of a breaker (62 of pattern B's 90 training stops alarm), at 18% more tokens than
`after_breakers`.

**The development reference under `every_version`** (the 300 development cases, success paths only): 247 repaired
by both audits, the same verdict as the first rule on 300 of 300; 6.09 actions a path against 4.24 (the rule
expert's replay 6.53), no look skipped.

**The full build** (the 2,550 training cases, at e95db77, 94 minutes on 26 workers; under
`output/minimal_env_20261121/`, not in git):

| | First build | Second build |
| --- | --- | --- |
| Paths: success / recovery / refusal / breakers-last | 2,550 / 1,855 / - / - | 2,550 / 1,855 / 912 / 635 |
| Of them repaired by both audits | 2,091 / 1,853 | 2,091 / 1,853 / 911 / 635 |
| Success path, mean actions | 4.26 | 6.11 |
| Look after an accepted breaker correction (success paths) | 300 of 1,600 | 1,600 of 1,600 |
| Rows | 16,077 | 33,181 (x2.06) |
| Prompt tokens, median / max / all | 9,221 / 14,948 / 149.0M | 12,327 / 16,477 / 396.8M (x2.66) |
| `CB_2R4_2R5` as a target | 41 of 2,109 breaker targets | 44 of 2,991 |

No look was skipped for the action budget and no healthy object was damaged.

**The selection** (`select_rows --repeat-breaker CB_2R4_2R5=3`, `output/minimal_env_20261121/selection_ieee14`):
the same 2,550 cases and the same split (2,398 training, 152 validation), 30,960 training rows (first selection
14,823) and 256 validation rows; the 305 `finish unresolved` rows on a passing estimate left out as before.  By
tool: `inspect` 14,544, `finish` 5,300, `correct_parameters` 4,761, `correct_topology` 2,895, `undo` 1,738,
`correct_measurements` 1,722.  The 39 rows that correct `CB_2R4_2R5` are written three times (78 more).  Prompt
tokens about 369M (137M); train sha256 aa0e7985..., validation 67ffcb58....

**What a pass on these rows costs** (my estimate): 7,740 optimizer steps at about 47,700 prompt tokens a step,
against the first pass's 3,706 at about 37,000 and 51 s a step over 49.5 hours: about 66 s a step, about 140 hours
for one pass from the base model.  The cell (`research/hpc/minimal_sft_20261121/`, prepared, not deployed) counts
75 s a step and would submit 16 links of 12 hours; the upload is 689 MB (the first pass's 270 MB took 13 minutes).
Cheaper ways, each a choice: `after_breakers` (about 16% fewer tokens, a rebuild of about 80 minutes), or the
first pass's adapter as the start (`sft1`) with fewer steps than one pass (not yet an option of the stage).
