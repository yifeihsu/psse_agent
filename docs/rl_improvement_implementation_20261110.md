# RL improvement workplan: review and implementation notes

Written 2026-11-10 against `docs/rl_improvement_workplan_20261110.md` (the workplan as received, unchanged).
Code base: branch `claude/rl-lifecycle-evidence-20261106` at `4b14d5e`, plus the updater repair that was
uncommitted in the study worktree.

**Where it stands.** All six tickets are built and tested on a CPU. Nothing has run on a GPU: the updater
calibration, the exact token audit, the learner's fork probe and the arms need the cluster. Nothing is committed,
deployed or submitted.

## 1. Review of the workplan

The plan's reading of the code holds at `4b14d5e`, and its numbers match the cluster record. Eight points change
how it should be run.

### What checks out

- **History limits.** The canonical observation keeps the last four events (`STUDY_POLICY_HISTORY_WINDOW`,
  `psse_env/dagger/evaluator.py`). The adapter then keeps at most eight events and 4,096 characters and drops the
  WLS tables of earlier events (`psse_env/dagger/dataset_builder.py`). Changing the adapter alone would miss the
  first limit, as the plan says.
- **Update.** One pass in two optimizer steps, a new AdamW per update, whole trajectories dealt to the steps so
  that a root's samples can part (`psse_env/rl/update.py`).
- **Configs.** The first arm's row bound was 1.0 and the sampled comparison arm's 25.0.
- **Candidate menu and target support.** Two lines in the ambiguous parameter menu; unsupported targets refused.
- **Scores on the 300-case development subset**, read on torch on 2026-11-10 at 14:38Z (bf16, bounded history,
  source `dcb6b38`):

  | Policy | Solved | Clean | Declines | True targets declined | Wrong targets kept |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Rule expert (route comparison on) | 188 | 184 | n/a | n/a | 44 |
  | Audit deciding | 200 | 189 | n/a | n/a | 0 |
  | S0, the cloned start | 182 | 168 | 5 | 2 | 47 |
  | S1 | 180 | 171 | 15 | 5 | 44 |
  | S2 | 181 | 169 | 10 | 4 | 47 |
  | S3 | 182 | 173 | 15 | 4 | 43 |

  The expert's and the audit's records are maps of case to result and hold no decline counts. S2 and S3 are new
  since the plan's snapshot. S3 solves as many cases as S0, so three updates gained nothing.
  The first arm's chain was cancelled on 2026-11-10 at 14:32Z; its run folder is kept. These six rows are the
  seed of the baseline register (section 2, T2).
- **The candidate audit** (the plan's "preliminary" table) reproduces from source records: the cloning behaviour's
  episodes on the 2,550 training roots hold 4,782 decisions on accepted candidates, and a line-parameter candidate
  is a true target in 0 of 111 in the classes whose faults are breakers alone, 1,990 of 2,086 elsewhere, and 567 of
  608 in the classes with a breaker and a line fault. Source:
  `output/compound_rl_20261106/cloning_ieee14_v2/episodes.jsonl` in the study worktree; command
  `python -m research.rl.fork_probe candidates --episodes <that file> --output <dir>/candidates`. These are counts
  of the candidates that behaviour proposed. A learner proposes others.

### What changes how the plan is run

1. **Two steps of a new optimizer still follow the first chunk.** Stage 0 says to start the calibration with two
   whole-root chunks. After the first step of a new AdamW every weight has moved by the learning rate against the
   sign of the first chunk's gradient; a second step can only lengthen or shorten that move. Keeping a root's
   samples together does not change this. A test shows a shared weight moving against the whole batch's gradient
   (`test_two_steps_of_a_new_optimizer_follow_the_first_step_against_the_batch`). So every arm takes one step
   whenever its optimizer is new, and two once the moments are carried. The calibration compares one step, two
   chunks and four chunks on one batch.
2. **The normalization leaves a shared prefix uncancelled.** Under `trajectory` normalization the steps a root's
   samples share are weighted by each sample's own length, so their advantages no longer cancel (16 to 21% of the
   shared weight was left on the first arm). The series uses `dapo`, which cancels exactly. This follows the
   study's recorded rule for new runs; it is a change from the first arm's objective and is listed in section 5.
3. **The first step's clip baseline depends on the GPU type.** The baseline is written once and a preempted update
   resumes on another card type. On the first arm that gave a mean gap of 0.024 and 0.064 on decision tokens at
   step one, where the gap should be zero. The series sets `first_step_baseline: own`: the first step clips against
   the row's own forward pass, and an update of one step runs no baseline pass at all. Later steps still clip
   against the persisted baseline, so the gap remains there. That part is not repaired.
4. **Complete history fits, at a cost.** As first rendered, the longest teacher episodes went over the old limit
   in several steps. Writing the WLS tables by column and pointing the newest event at the current tool output
   loses nothing and leaves 2 steps of 1 episode in 140 over it, with a longest prompt of 33,737 tokens. Episode
   input tokens rise from a median of 64,038 to 92,367 (1.44 times) and from 174,221 to 391,439 at the 95th
   percentile (2.2 times).
5. **The 32,768-token limit was ours, not the model's.** `google/gemma-4-12B-it` has 262,144 positions. The series
   runs at 49,152 tokens (section 3).
6. **A replay from a stored observation did not reproduce the live prompt.** The bounded renderer numbers case
   aliases, and cuts long values, in the order it meets a mapping's keys. A stored record has its keys sorted; the
   running environment does not. 142 of 395 bounded views differed, and every replayed fork branch was invalid
   until the fork signature sorted keys. The update is not affected: it re-renders the view the policy was given.
7. **The decision credit is small where the policy is confident.** The fork-local advantage of a sampled choice is
   (1 − p) × (value of that choice − value of the other). Where the policy keeps a wrong target with p near 1,
   keeping it earns almost no negative credit; the signal comes from the rare sampled declines.
8. **"True candidate after an earlier decline" will be short.** S0 declines 5 of 562 accepted candidates under
   greedy decoding, so few such states exist. The probe records the shortfall, as the plan asks.

Not rechecked here: the plan's count of 265 identical tool sequences between S0 and S1.

## 2. What was built

| Ticket | What it gives | Main files | Tests |
| --- | --- | --- | ---: |
| T1 updater continuity | whole-root steps, carried optimizer, one step when the optimizer is new, row accounting, batch digest, step diagnostics, stop conditions | `psse_env/rl/update.py`, `psse_env/rl/train.py`, `psse_env/rl/integrity.py` | 27, and 15 from the study's repair |
| T2 matched run manifests | one base config and four matched arms, reserved roots, baseline register, provenance record, job scripts | `research/hpc/rl_improvement_20261110/`, `research/rl/baseline_register.py` | 16 |
| T3 complete history mode | versioned observation mode, complete renderer, refusal instead of truncation, token audit | `psse_env/dagger/observation_modes.py`, `dataset_builder.py`, `evaluator.py`, `research_policy_factory.py`, `research/rl/history_audit.py` | 17 |
| T4 fork probe | frozen fork sample, replayed and forced branches, paired report, decision panel | `psse_env/rl/forks.py`, `research/rl/fork_probe.py` | 20 |
| T5 decision credit | continuation phase, fork-local advantage, provenance checks | `psse_env/rl/decision_credit.py`, `psse_env/rl/rows.py`, `psse_env/rl/train.py` | with T1 and T4 |
| T6 evaluation report | paired scorecard, recovery chain, costs, promotion and stop rules | `research/rl/scorecard.py` | 5 |
| Prompt limit | the run's own limit, capacity read, memory refusal | `psse_env/rl/train.py`, `research/rl/prompt_capacity.py` | 12 |

Every new option is off by default, so a recorded config runs as before.

### T1 Updater

- **Repair** (from the study worktree): `chunk_by: "group"` keeps a root's samples in one optimizer step;
  `carry_optimizer: true` leaves the optimizer state in the iteration directory and the next update continues it.
- **`fresh_optimizer_steps`**: the steps an update takes when its optimizer has taken none. A resumed update plans
  what the interrupted one planned.
- **Accounting.** The report says how the rows were dealt (`row_accounting`: every row in exactly one step, rows
  by kind). A deal that repeats or drops a row is refused before anything is trained.
- **Batch digest.** A checkpoint is bound to its batch of rows; a resume on another batch is refused, for the
  partial clip baseline as well.
- **Step diagnostics** (`step_diagnostics`): per optimizer step, the rows the row bound clipped, the rows that
  carry the step, how far the weights moved in units of the learning rate, and the probability the model gives the
  lifecycle choices on the batch's labelled decisions. With `diagnostic_components` the step's gradient is also
  split by loss term (policy, KL, lifecycle, supervised); that costs one more backward pass per further term, so
  the arms leave it off and the calibration runs turn it on. The diagnostics do not change the weights.
- **`first_step_baseline: "own"`**: point 3 of the review.
- **Stop conditions** (`psse_env/rl/integrity.py`): a nonfinite loss or gradient, a re-rendered prompt that is not
  the sampled one, a sampled row without its sampling record, a row cut at the training length or at generation,
  a row in no step or in two, more steps than planned, a carried optimizer that counts other steps. With
  `stop_on_integrity_failure` the driver writes a marker beside the update report and refuses the phase.
- **Verified on the small deterministic model:** training on equals saving and reloading, across optimizer steps
  and across iterations; a preemption inside an update changes nothing (four cut points); the dealt steps add up
  to the objective and gradient of the whole batch before the optimizer is applied; a root with contrasting
  samples, an invalid sampled output and a preemption trains every row once.

### T3 Complete observable history

- **Two named modes.** `bounded_history_v1` is what every recorded cell ran. `complete_history_v1` hands the
  policy every transition of the episode.
- **Built from the sanitized policy transition stream**, so the truth, the class, the audit's labels and the
  evaluator's verdicts stay out. A test changes only audit metadata and gets the same prompt.
- **State binding.** Each event names the case it ran on by a stable alias: `active`, `candidate`, or `sN` in order
  of first appearance. A list of candidate experiments says whether each was committed, rolled back or is open.
- **Nothing is removed for size**: no event limit, no table removal, no character budget. The system prompt gains
  one paragraph that says the record is complete.
- **A prompt that does not fit is refused** (`PromptLimitError`), recorded as an invalid action and counted in the
  rollout summary. The bounded mode still cuts from the left; the row builder counts those cuts and the integrity
  check stops on them.
- **The bounded mode reproduces its prompts**: 192 policy steps of 14 teacher episodes are identical before and
  after the change (views, aliases, system prompts), and a golden digest test holds it.
- **The acceptance checks of the plan are tests**: an early WLS table and a candidate edit visible after more than
  four actions; decline keeps the declined candidate's evidence and restores the parent as current; commit rebinds
  the aliases without erasing; hidden metadata; the old mode's prompts; no size limit.

### T4 Fork probe

- **A fork** is a decision the learner sampled on a candidate the verifier accepted, where commit and decline are
  both legal.
- **`select`** takes one fork per root by the plan's kinds, with quotas (32 training, 16 validation), a share of
  roots the rule expert fails, and related roots kept in one split. The rules and seed are in the manifest. A kind
  the stored states cannot fill is a recorded shortfall.
- **`branch`** replays the fork's prefix through the real environment, forces the choice and lets the policy
  continue under the budget the episode had left. A branch counts only if the replay showed the policy what the
  recorded episode showed it at every step. The forced step is recorded as an intervention.
- **`views` and `decide`**: what the policy is shown at each fork in each mode, and its commit and decline
  probabilities there (48 states, two views).
- **`report`**: the value of each choice by kind, split and mode; what followed a forced choice (was a true target
  offered, chosen, accepted, kept, was the case solved); the learner's decline probability under both modes; the
  screening count.
- **`panel`**: the same forks scored before and after an update, which is stage 0's fixed decision panel.
- **`reference`** and **`candidates`**: the rule expert's own result on the probe's roots, and the candidate audit.

### T5 Decision and continuation credit

- **A new phase.** A run with `decision_credit` goes rollout, continuation, update, evaluate. The continuation
  phase picks up to 32 sampled decision states (new roots first, by kind), runs both choices with the current
  policy, twice each, and writes `decision_values.json`.
- **The objective.** For a covered decision, Q(c) is the mean return after choice c: the branches that forced it
  and, for the choice a sample took itself, that sample's own return. The baseline is V = Σ p(c) Q(c) over the two
  choices, taken from the forced branches alone, so it does not depend on the action that was sampled. The sampled
  row is credited with weight × (Q(a) − V). Its episode advantage is set to zero and its supervised membership row
  is dropped. Every other row keeps the current objective, and the KL term to S0 stays.
- **Forced branches are never training rows.** They only give values.
- **Provenance.** Values written for another iteration, adapter or observation mode are refused. A value for
  another action than the row's is refused. A credited row without its sampling record is not trained.

### T6 Evaluation report

`research/rl/scorecard.py` reads the driver's episode records, a compound-study episode file or a map of root to
result. The primary number is paired: cases only the learner solves less cases only the reference solves, both
counts shown, with an exact two-sided sign test. Beside it: clean solves by class, expert-failed cases recovered,
declines of true targets (first and later candidates, classes with a breaker and a line fault), wrong targets kept,
healthy components corrupted, wrong declines that were followed by a kept true correction and a solve, process
failures, tool calls on the cases both solve, tokens and time. `promotion_check` and `stop_rule` are the plan's
pilot rules as code.

### T2 The run cell

`research/hpc/rl_improvement_20261110/README.md` describes it and gives the commands in order. In short:

- **`series.json`** holds one base config and the arms' overrides; `make_configs.py` writes the configs and checks
  them. B differs from A in the observation mode alone, C from A in the decision credit alone, D from B in the
  decision credit and from C in the observation mode. A test holds this.
- **Arm A is the first arm with the repaired update.** Same roots, same seeds, so its first rollout batch is the
  first arm's.
- **Reserved roots.** 64 training roots (the series' own stratified draw of iteration 30) that no arm may draw:
  the driver refuses an iteration that would. They are the probe's validation split and the decision panel. The
  probe's training split is the draw of iterations 28 and 29 (128 roots).
- **Baseline register** (`research/rl/baseline_register.py`): each finished evaluation with its checkpoint, suite
  digest, action limit, precision, observation mode, prompt limit, source commit, result and file digests.
  `compare` says whether two entries can be compared and what they differ in.
- **Provenance per job**: source commit, config digest, input digests, and whether each input is the one the
  series was planned on. A difference is reported, not refused.
- **Calibration** (`calibration.py`): a run that starts at the update of a fixed rollout batch, and a report that
  puts the runs side by side.
- **Job scripts**: `rl.sbatch` (with the continuation phase, the integrity stop and a capacity task),
  `fork_probe.sbatch`, `submit_arm.sh`, `deploy_from_windows.sh`.

## 3. The prompt limit

The series runs every arm at **49,152 tokens** (`max_input_tokens` and `max_length`). The earlier cells ran at
32,768.

- **The model allows far more.** `max_position_embeddings` is 262,144 for `google/gemma-4-12B-it` at revision
  `707f0a3b` (its `config.json` on torch, and the header of a local GGUF copy).
- **The cards do not.** On the first arm (bounded prompts of at most about 13,000 tokens) the update process peaked
  at 34.1 to 37.3 GB and two rollout workers at 58.5 to 63.5 GB together. That is about 24 GB of weights and about
  1 MB per prompt token in the update. The cards are 80 GB (A100, H100), 96 GB (RTX PRO 6000) and 141 GB (H200).
- **49,152 tokens extrapolate to about 73 GB in the update**, which is inside an 80 GB card with little room. It
  is an extrapolation from one point. `research/rl/prompt_capacity.py` measures a generation and an update row at
  each length on the card a job gets (`TASK=capacity` in `rl.sbatch`); run it before the arms.
- **It covers what has been seen.** The longest complete-history prompt of the 140 teacher episodes is 33,737
  tokens, in an episode of 38 actions. The steepest episode projected to the 40-action limit is about 41,500. At the new limit
  no step of those episodes is refused; at the old one 2 steps of 1 episode were.
- **The limit is the config's now.** `max_input_tokens` is a field of the run config. Every phase sets the
  policy's limit from it and records it in its summary. A limit beyond the model's window is refused when the
  model's config is on the machine.
- **Two safeguards for long prompts on a shared card.** After a complete-history prompt of 16,384 tokens or more
  the worker returns its cached device memory, because two workers share a card and the allocator would otherwise
  hold each one's peak. A complete-history prompt that still finds no room is tried again after 10 and 30 seconds
  and then refused and counted, as an over-long prompt is. The same episode's later prompts, which are longer, are
  tried once without the wait. The bounded mode is unchanged.
- **Not covered:** an update row that does not fit ends the update process with the device's error. The next job
  of the chain retries, possibly on a larger card.

The bounded mode's own limits (four events, eight events and 4,096 characters, bounded WLS tables) are unchanged:
they are what arm A is.

## 4. Measurements made here

All on a CPU, with the rule expert where a policy is needed.

**Token audit, teacher episodes** (140 episodes, 10 per class, 1,794 policy steps; a tokenizer file of the model
family, which counts 2 to 3% above the real prompt):

| Prompt tokens | Median | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Bounded history | 5,618 | 8,738 | 11,266 |
| Complete history | 8,963 | 19,602 | 33,737 |

Fidelity passed in every episode: every transition present, each event's evidence equal to what was shown when it
was the latest, aliases stable, every candidate's fate listed. Reports: `output/rl_improvement_20261110/history_audit_teacher_dev/`
(old limit) and `history_audit_teacher_dev_limit_48896/` (new limit). The plan's gate is the same audit on the
learner's own rollouts with the model's processor; that needs the smoke's rollouts.

**Fork probe, wiring check** (the rule expert rolled out and continuing, on 84 and 42 roots of the development
subset; one continuation, since the expert is deterministic). This checks the harness on the real environment. It
is the expert's own continuation value, not the learner's.

- 48 forks, 192 branches, all valid in both modes. The two modes give the same branches, as they must: the expert
  reads the bounded window either way.
- Coverage shortfalls: other wrong targets 4 of 8 and 2 of 4; true candidates after a decline 0 of 4 and 0 of 2
  (the expert never declines).
- By kind:

  | Kind of fork | Forks | Decline better | Solved after commit | Solved after decline |
  | --- | ---: | ---: | ---: | ---: |
  | Wrong line candidate, faults are breakers alone | 11 | 11 | 0 | 7 |
  | Other wrong target | 6 | 4 | 0 | 1 |
  | True line candidate, breaker and line fault | 19 | 1 | 13 | 2 |
  | True first candidate | 12 | 0 | 10 | 0 |

- After a forced decline of a wrong target (17 continuations): a true target was offered in 7, corrected next in
  6, accepted and kept in 6; 8 were solved.
- 9 distinct roots the expert fails were solved by a continuation (6 training, 3 validation), of 25 probed.
- In 3 of 48 forks the choice with the better continuation is not the one the membership label names.
- Of 35 line-parameter candidates one used the pair refit; none was labelled true while it changed a healthy line.

Report: `output/rl_improvement_20261110/fork_probe_wiring/report.md`.

**The driver's phases on the real environment** (rule expert, complete history, 6 roots, 12 episodes): the
rollout, continuation and update phases ran in order, no prompt was cut or refused, and each summary carries the
prompt limit. The expert's episodes have no sampling records, so the continuation phase found no state to branch
and the update carried the adapter over. The continuation phase with real branches is tested on the toy loop only;
`smoke_complete_credit.json` is its first real run.

**Bounded mode unchanged:** 192 policy steps of 14 teacher episodes give identical prompts before and after.

## 5. What only the cluster can show, and what is open

Not verified, because it needs the model:

- the complete-history prompt through the model's processor (exact token counts), and S0's behaviour under it;
- that the supervised targets render under the complete history (`target_renderer.refused == 0`) and that the
  update's first step is on policy there;
- memory and time at the new prompt limit, for a generation, an update row, and two workers on one card;
- the decision-token probabilities the continuation phase reads from the sampling records;
- every learner number: the calibration, the fork probe with learner continuations, the arms.

Not built:

- stage 5 (three seeds and an untouched test set; the configs carry one seed);
- the conditional work on candidate availability and additional evidence;
- the policy-gradient-off control;
- a clip baseline taken in the stint that trains the row, for steps after the first.

Open decisions:

1. **Commit.** Nothing is committed. This worktree also holds the study worktree's uncommitted updater repair, so
   the two overlap; commit one of them, not both.
2. **The update settings of the base**: `dapo` normalization, one step when the optimizer is new, the row's own
   forward as the first step's clip baseline. The first two follow the study's recorded rule for new runs.
3. **The prompt limit**: 49,152 tokens now, or the figure the capacity read supports.
4. **The probe's expert reference**: the rule expert with its route comparison on, run on the probe's roots.
5. **Cluster work**: deploy, the two smokes, the capacity read, then the calibration and the probe. Each needs
   your word.

A rough size of the cluster work, from the first arm's 54.9 GPU-hours for a baseline and three iterations: arm A
about 40 GPU-hours for its baseline and two iterations; arm B more, in proportion to its input tokens (1.4 to 2
times); each calibration update about 6 to 10; the probe about as much as one iteration's rollouts twice. The
smoke measures the real figures.

## 6. Tests

The nine new test files hold 97 tests; with the study's additions to `test_update.py` and `test_train.py`, 112
tests are new. Two of the repository's cell tests share the file name `test_cell.py` with the new cell's, so the
three are run one directory at a time:

```
python -m pytest psse_env research tests --ignore=research/hpc/compound_rl_20261107/test_cell.py --ignore=research/hpc/rl_improvement_20261110/test_cell.py
python -m pytest research/hpc/compound_rl_20261107/test_cell.py
python -m pytest research/hpc/rl_improvement_20261110/test_cell.py
```

Result on 2026-11-10 in this worktree: 2,785 passed, 15 skipped, 11 failed in 14 minutes 49 seconds; the two cell
tests passed (3 and 13).

- **10 of the 11 failures are not from this work.** They are in `psse_env/dagger/test_suite_builder.py` (the
  release builder's pins on the Python and package versions) and fail the same way at `4b14d5e` in the study
  worktree, which holds the updater repair and nothing else of this work.
- **The 11th is an OpenDSS solve that did not converge**
  (`tests/test_hif_operating_point_regulation.py::test_injected_hif_candidate_keeps_regulation`). Other jobs were
  running on the machine during the suite. Run alone it passed three times of three, and this work does not touch
  the code it runs.
- **The 15 skips** are tests that replay local data this worktree does not hold (a frozen audit, a development
  suite, the DAgger corpora, a reviewed canary).
- After the last edits the RL, research and `psse_env/dagger` tests were run again: 694 passed, 2 skipped (the
  suite-builder file left out).
