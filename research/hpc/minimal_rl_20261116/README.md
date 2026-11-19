# RL in the minimal environment (deployed 2026-11-16)

Runs `research/minimal_env/rl_train.py` on the cluster: sampled episodes of `psse_env/minimal`, the outcome reward,
one on-policy update per iteration with a KL penalty to the supervised model, and the greedy evaluation on the 300
development cases. What an iteration does and why its settings are what they are:
`docs/minimal_env_20261115.md`, section 12.

**State: the run's first iteration finished on 2026-11-18 at 22:16 UTC (`mrl-run` 23608433, source e82026b)
and is read: 202 of the 247 cases a truth-built sequence repairs, against the supervised model's 198 (rule
expert 184), 5 cases gained and 1 lost, a split that one iteration cannot tell from chance ("The run, submitted
2026-11-18").** The smoke passed before it (`mrl-smoke` 23435096, 2026-11-17, 22 minutes on an H200): every phase
ran once and every check of the update held. The run took the supervised model's evaluation as its baseline: it
was made under this run's settings (compared with its recorded settings under the cell's environment). Two
things the smoke showed about the run itself are under "What the smoke showed that the run should know".

**The focused run is deployed (2026-11-19, source 2d4919d), and its training pass was submitted on 2026-11-19 at
19:17 UTC on the user's word (`mrl-train-pass` 23746858, 23746859, 23746861, 23746863; `mrl-train-pass-close`
23746865). The focused smoke and the focused run are not submitted: "The focused run" below.** It draws from the
training cases a truth-built sequence repairs and the supervised model does not, and teaches that sequence where
no sample repairs. The deploy waited for the plain run's links to leave the queue, on the user's word of
2026-11-18 ("Yes, after it ends").

## What it runs

| Job | What |
| --- | --- |
| `mrl-smoke` | one iteration on 6 training cases by 2 samples and 8 development cases, baseline included; 4-hour limit |
| `mrl-run`, a chain | the run, to the iteration count named at submission; each link goes on from the run's manifest |
| `mrl-train-pass`, `mrl-focus-smoke`, `mrl-focus` | the focused run's jobs ("The focused run") |

One stage serves both (`stage_rl.sbatch`). It runs whatever phase the manifest is at, one process per phase, until
the target is reached or the time runs out. On a card of 70 GB or more the rollout and the evaluations run as two
processes, each taking every second case; the update is always one process. Every phase resumes: one file per
sampled case and per evaluated case, and the update checkpoints itself every 32 rows.

Before any GPU work each link runs the driver's `--check`: the suites load, the first iteration's cases draw by
class, the adapter is there.

## Where the run starts

- **The adapter of the supervised pass** (`/scratch/yx3882/minimal_sft_20261115/out/pass_base/training/lora`),
  read in place. It is also the reference of the KL penalty in every iteration. The run refuses to start before
  that pass has written its receipt.
- **The baseline is the supervised cell's own evaluation** of that adapter on the 300 cases, once it is finished:
  the run takes its episodes instead of running them again, so the supervised model has one number. The driver
  takes it only if it was made for the same adapter under the same model, precision, episode length, table window
  and prompt limits, and refuses it otherwise; while that evaluation is not finished the baseline is run here.
- **The smoke starts from a copy** of the pass's adapter or, while the pass is running, of its newest complete
  checkpoint (the pass removes old checkpoints, so the cell keeps its own copy under `data/smoke_adapter`, with
  the checkpoint it came from in `SOURCE.txt`). Its baseline is run by the smoke itself.

Nothing in the supervised cell's folders is written to. This cell's source is a clone of that cell's source, moved
to the deployed commit; the supervised cell's jobs keep running from their own.

## Settings of the run (`rl_ieee14.json`)

The config is the driver's example (`research/minimal_env/rl_config_example.json`) on the cluster's paths; a test
holds the two equal. The smoke's config differs from it in size and start alone.

| Setting | Value | Whose |
| --- | --- | --- |
| Cases per iteration, samples per case | 64, 4 | the first arm's values, not decided anew |
| Draw | by class, the first arm's counts (8 to 2 per class) | the first arm's |
| Learning rate, KL weight | 1e-5, 0.04 | the first arm's |
| Optimizer steps per update | 1, the optimizer carried across updates, no clip | what the first arm measured (section 12) |
| Advantage | reward less the case's mean, not scaled | mine (section 12) |
| Rows | whole groups only, prompt digest required, no cut prompt, up to 24 cases' rows for the KL term alone | the first arm's |
| Episode | 16 actions, tables of two versions, bf16, prompts to 32,768 tokens | the supervised cell's |
| Training and development cases | the suites of 2,550 and 300 cases the earlier RL cells used | the same files the demonstrations were built from |

The first four rows are yours to change before the deploy: edit both config files.

There is no stop rule in the cell. A run goes to the iteration count it was submitted with, and `run 5` after
`run 3` takes it two further. An update that fails an integrity check stops the run where it is and leaves
`integrity_failure.json` beside its report; later links then exit at once.

## Estimates (my arithmetic from the smoke's rates)

Per iteration of 64 cases by 4 samples on an H200 with two processes on the card:

- rollout, 256 episodes of 4 to 9 actions at 4.7 s a step: 40 to 90 minutes;
- update at 9.2 s a row: 1.3 to 3.3 hours if 20 to 64 of the 64 cases are informative (about five rows a sampled
  episode, plus up to 120 rows for the KL term), under an hour if few are;
- evaluation, 300 cases of 5 to 9 actions at 4.9 s a step: 1 to 2 hours.

Together 3 to 6.5 GPU-hours an iteration on an H200. The supervised pass ran 1.4 times slower on an RTX PRO 6000;
at that ratio 4 to 9 hours there. Before the smoke I had estimated 8 to 14. The update rate is from 27 rows, and
no card but an H200 has run any of it.

`submit_cell.sh` still counts 14 hours an iteration and 11 hours of each 12-hour link, plus a spare: 3 links for
one iteration, 4 for two, 5 for three. That is more links than the rates above need; a link that finds the target
reached exits at once.

The smoke took 22 minutes; I had estimated an hour.

## Order of work

```bash
bash research/hpc/minimal_rl_20261116/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch bash -s -- smoke < research/hpc/minimal_rl_20261116/submit_cell.sh
```

```bash
wsl -- ssh torch bash -s < research/hpc/minimal_rl_20261116/status_cell.sh
```

```bash
wsl -- ssh torch bash -s -- run 3 < research/hpc/minimal_rl_20261116/submit_cell.sh
```

The `3` is an example: the count is named at submission and has no default.

## The deploy of 2026-11-16

`deploy_from_windows.sh` cloned the supervised cell's source into `/scratch/yx3882/minimal_rl_20261116/source` and
moved the clone to e82026b with a bundle of 60,956 bytes. It found both suites with the planned checksums, the
reference and the job's Python in place, and the supervised pass and its evaluation not finished. It submitted
nothing.

Read on the cluster afterwards (15:50 UTC):

- the source is at e82026b with no tracked change; the supervised cell's source is still at 9c80d9a, under its
  running jobs;
- the four scripts parse under the cluster's bash and hold no carriage return;
- the suites and the reference both configs name exist; the run's adapter and baseline folder do not yet (the pass
  is running), and the smoke's adapter is copied by the smoke itself;
- `newest_complete_checkpoint` names `checkpoint-816` of the running pass (3,706 steps in all);
- the driver imports under the job's Python (3.12.13) with the cell's environment;
- `status_cell.sh` ran there for the first time, on the empty cell: source, no receipt, no job.

Commits on the branch after e82026b up to 0960e22 were records alone (the READMEs and the notes). 2d4919d added
the focused run in code; it was deployed on 2026-11-19 ("The focused run").

## The smoke of 2026-11-17

Submitted at 05:08 UTC on the user's word, one job, `mrl-smoke` 23435096, from the source at e82026b. Read just
before: no job of this cell queued, the cell's `out`, `data` and `logs` empty, and the supervised pass running
with `checkpoint-1648` (of 3,706 steps) its newest complete one. The smoke copies whichever checkpoint is newest
when it starts and writes its name to `data/smoke_adapter/SOURCE.txt`.

**Completed in 21 min 52 s** on one H200 (gh104), started 6 minutes after submission, no preemption, no traceback,
last line `mrl-smoke-complete`. It copied `checkpoint-1648`. The manifest stands at iteration 1, phase rollout.

Against the list below ("What the smoke has to show before a run"):

| Item | Result |
| --- | --- |
| The dry check, first run on the cluster | ok: 2,550 training cases, 6 drawn, 8 development cases, adapter found |
| Every sampled step carries its token probabilities | 45 of 45 steps; rows without a record 0 |
| Steps whose tokens are all above 0.99 | 24 of 45 (53%) |
| Every stored step renders back to its sampled prompt | 27 of 27 rows: no length or digest mismatch, no cut prompt |
| Sampler against the update's first pass | mean gap 0.0006 in log-probability; 0.010 on the 24 uncertain tokens, largest 0.116; no token beyond the limit (the batch is refused above a mean of 0.2) |
| Integrity of the update | ok, no failure, no warning; each row used once; one optimizer step |
| Card memory | two processes together 63.4 GB (baseline), 63.2 GB (rollout), 63.0 GB (evaluation); the update alone 36.5 GB, prompts up to 12,203 tokens |
| One pass through the phases | baseline 6 min 19 s, rollout 3 min 43 s, update 5 min 38 s, evaluation 5 min, loading included |

Rates: 4.9 s a generated step (4.7 s when sampling); a sampled episode 17.6 s (3.75 actions on average); an
evaluated episode 50 s (9.4 actions); the update 9.2 s a row (27 rows in 249 s). Prompts ran from 5,221 to 12,694
tokens. Loading the model took one to two minutes a process.

**What the episodes were.** The six training cases were of six classes, the eight development cases all of one
(`cb2_dangling`, two breaker errors: `development_limit` takes the suite's first eight, and the suite is ordered
by class). The two sets cannot be compared with each other.

- Rollout, 12 sampled episodes: 8 solved, 4 declared resolved with an error left. Five of the six cases gave the
  same episode twice. The sampled episodes' own probabilities were 0.85, 0.81, 0.89, 0.94 and 0.56 for those five,
  and 0.45 and 0.04 for the two episodes of the sixth, which differed by one refused call (a breaker name with a
  part missing). No case had one repaired and one unrepaired sample.
- So the update had one informative case of six: 9 policy rows with advantages of plus and minus 0.015 (the cost
  of one action and one refused call), 18 rows for the KL term, gradient norm 0.08, policy loss -0.0002. The KL
  term was exactly 0, as it must be in the first iteration (the reference is the adapter the update starts from).
- Baseline, greedy, 8 cases: 1 solved. The first breaker correction was a true error in 6 of 8; the second breaker
  error was found in 1. After the first correction the model went on to parameter fits and meter repairs, and in 6
  of 8 it declared the case resolved (5 of them wrongly). 17 refused calls, all parameter fits that found nothing:
  10 of them one call repeated in one episode, which ran out its 16 actions; 2 a fit that failed inside the tool
  (`parameter scan 0 must contain 125 finite values`, on lines the model had just cut off), which the environment
  reported to the model as a refused call.
- Evaluation after the update, greedy, the same 8 cases: 0 solved. Five episodes are the baseline's step for step;
  three differ, from step 3, 3 and 7, and the one solved case is lost.

This is the supervised model 44% through its pass, on eight cases of one class: it is not a reading of the
supervised model.

## The run, submitted 2026-11-18

`submit_cell.sh run 1` at 17:12:55 UTC on the user's word, source e82026b: one iteration, asked for first so that
the two things the smoke left open are read with the finished model before more is spent.

| Job | Ids | Starts after |
| --- | --- | --- |
| `mrl-run`, 3 links | 23608433, 23608439, 23608441 | each after the one before it |

Read just before: the source at e82026b with no tracked change; no job of this user queued; `out/ieee14` not
there; the supervised pass's receipt and adapter present; the supervised evaluation present with its 300
episodes; both suites and the reference in place.

What it is read for:

- the rollout: of the 64 cases, how many have a repaired and an unrepaired sample among their four
  (`groups_with_repaired_and_unrepaired_samples`), how many differ in reward at all, and the share of steps whose
  tokens are all above 0.99;
- the baseline taken from the supervised cell's evaluation (198 of 300), the first time that path runs on the
  cluster;
- the update at full size: rows, seconds a row, memory, the gap between the sampler and the first pass;
- the evaluation after the first update: cases gained and lost against the 198.

**Outcome, read 2026-11-19 at 08:06 UTC: the iteration solves 202 of the 247 cases a truth-built sequence
repairs, against the supervised model's 198** (rule expert 184): 5 cases gained, 1 lost.

| | the supervised model | after one update |
| --- | --- | --- |
| solved, of the 247 | 198 | 202 |
| of the 184 the rule expert's episodes solve | 161 | 165 |
| of the 63 only the truth-built sequence repairs | 37 | 37 |
| declared resolved on an unsolved case | 72 | 73 |
| declared unresolved | 27 | 24 |
| out of actions (16) | 3 | 1 |
| refused calls (unknown object, tool failure) | 88 (30, 58) | 67 (15, 52) |
| damaged healthy objects (in episodes) | 51 (27) | 39 (26) |
| mean actions | 4.45 | 4.36 |

- **The jobs.** The first link started on gr103 (RTX PRO 6000) at 17:13 UTC, was preempted at 21:05 in the
  evaluation, came back on gr104 (RTX PRO 6000) at 21:13 and went on from the manifest with the evaluation; it
  ended at 22:16. The two later links found the target reached and ran nothing (22:39 and 22:43, about a minute
  each). The baseline took 10 s (taken from the supervised cell's evaluation, the first time that path ran on the
  cluster), the rollout 1 h 12 min, the update 2.28 h, the evaluation about 1 h 20 min of card time across the
  preemption. Peak card memory 66,770 MiB, mean utilization 93%.
- **The rollout.** 64 cases by 4 samples; 167 of the 256 episodes solved. 36 cases had every sample repaired;
  9 had a repaired and an unrepaired sample, the only cases an update can learn a repair from; 19 had no repaired
  sample, 11 of them cases a truth-built sequence repairs and 8 cases none repairs. 30 cases sampled more than
  one distinct episode and 23 differed in reward. 55.6% of the 1,195 steps had every token above 0.99. Refused
  calls: 69 unknown objects, 61 tool failures.
- **The update.** One step on 629 rows (535 sampled rows of the 23 cases with a reward spread, 94 preservation
  rows) in 2.28 h. Policy loss -0.0085, KL 0 (the reference is the start), gradient norm 1.14. The first pass
  against the sampler: mean |log ratio| 0.0014, 0.019 on the 704 uncertain tokens, largest 1.29; no row without a
  record; integrity ok.
- **The cases that moved.** Gained: two two-breaker cases (`r0_37d75bce0a1b` cb2_split, `r0_566f34094a8a`
  cb2_mixed) where the supervised model sent one topology correction fifteen times, refused each time as an
  unknown object, until its actions ran out (these two cases held all 30 of its unknown-object refusals); the
  updated model's two corrections are accepted, with an inspection between them, and it finishes in five
  actions. And three cases where the supervised model's corrections damaged healthy objects: `r0_a04d7fb1791e`
  (rx2_adjacent; a measurement correction, nine objects damaged, declared unresolved), `r0_c35a1e4bfa40`
  (rx2_distant; a measurement correction after the two parameter corrections, two damaged, one error left) and
  `r0_c2b40c5df2b0` (control_meter; both errors repaired, one healthy object damaged on the way, seven actions);
  the updated model makes the two corrections each case needs and nothing else. Lost: `r0_ee184f6d4f8c`
  (rx2_distant, one of the cases only the truth-built sequence repairs), where the updated model declares
  resolved after the first of its two parameter corrections. Of the other 294 cases, 274 repeated their episodes
  exactly and 20 changed their actions but not their outcome.
- **The loop moved, it did not go.** All 15 unknown-object refusals after the update are one case,
  `r0_ab11377c26a5` (unsolved before and after), where one refused topology correction is now sent fifteen times
  until the actions run out; before the update that case had two refused parameter corrections.
- **How far this goes.** Both evaluations were greedy, in bf16, on RTX PRO 6000 cards, so the card-type variation
  of the earlier cells does not enter: the six cases moved because the adapter moved. But five gained against one
  lost is a split two equally likely directions give about one time in five (two-sided sign test on the six
  cases, p = 0.22), and two of the five are one loop in two cases. One iteration does not show that the update
  helps.

## The focused run (built 2026-11-18, deployed 2026-11-19)

Asked for by the user on 2026-11-18, after the supervised model's evaluation: results are read against the 247
development cases a truth-built sequence repairs, the model fails 49 of them, and the run should aim at such
cases. The plain run above cannot: its draw spends most of an iteration on cases the model already repairs or
that no sequence repairs, and on the cases the model fails its own samples seldom hold a repair to learn from (the
smoke: five of six cases sampled one episode twice).

**What it changes, against the plain run.** The start, the baseline, the reward, the learning rate, the KL
weight, the one optimizer step and the row checks are the plain run's (`rl_focus.json` differs from
`rl_ieee14.json` in `focus`, the draw, the samples per case, the seed and the output folder; a test holds that).

1. **The draw.** Only the 2,093 training cases a truth-built sequence repairs (`train_truth_sequences.json`,
   built from the local demonstration build by `research.minimal_env.truth_sequences`; the other 457 training
   cases are never drawn). An iteration takes 24 of the cases the supervised model leaves unrepaired and 8 of
   the ones it repairs. Which is which is read from the supervised model's own greedy pass over the 2,093 cases
   (`mrl-train-pass`); the driver refuses a pass of another adapter or under other settings.
2. **Eight samples a case** instead of four: 32 cases by 8 is the plain run's 256 sampled episodes.
3. **A truth-built example where no sample repairs** (the user's "Yes, include them"). When none of a drawn
   case's eight samples is repaired, the rollout's closing call replays that case's sequence in the environment
   and keeps it under `iteration_NNN/demonstrations`, beside the groups and never in one. The update trains each
   of its actions as a supervised row: the action, rendered as the supervised pass taught it, on the prompt the
   model would have read at that point. A row's weight is `demonstration_weight` (1.0) times how far the
   sequence's reward lies above the mean reward of the case's samples. A case with a repaired sample gets no
   example: its group has its own contrast.

**The third change is imitation, not RL.** Its rows have no sampling record and no advantage, and take no part in
the KL term or the on-policy check; they are the truth-built sequence, taught on exactly the cases the model
fails. The sampled rows of those cases stay in the update for the KL term (`rows.preservation`), and the driver
refuses a config that would train examples with no sampled row beside them.

**Mine to have chosen, yours to change** (`rl_focus.json`, and `rl_focus_config_example.json` beside the driver;
a test holds the two equal): 24 and 8 cases, 8 samples, the weight 1.0, giving an example only when no sample
repairs, and weighting it by its reward above the samples' mean.

| Job | What |
| --- | --- |
| `mrl-train-pass`, 4 at once, then `mrl-train-pass-close` | the supervised model, greedy, on the 2,093 cases: each job two of eight shards, one file per case; the closing job writes `out/train_pass/evaluation.json` |
| `mrl-focus-smoke` | one focused iteration: 6 unrepaired and 2 repaired cases by 2 samples, 8 development cases; 4-hour limit |
| `mrl-focus`, a chain | the focused run, to the iteration count named at submission |

Both focused jobs refuse to start before the training pass has written its evaluation.

**Estimates (my arithmetic from measured rates).** The training pass: 2,093 episodes at the 30.5 s an episode of
the supervised model's development evaluation (two processes on an RTX PRO 6000) is about 9 GPU-hours, a little
over two hours of wall time if the four parts start together. A focused iteration: about the plain run's, with
up to about 100 more update rows for the examples (24 cases by 4.1 actions a sequence; about 15 minutes at the
smoke's 9.2 s a row).

**Order of work**, each step on the user's word. The deploy refuses while a job of this cell is queued, so it
waits for the plain run's three jobs to leave the queue.

```bash
bash research/hpc/minimal_rl_20261116/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch bash -s -- train-pass < research/hpc/minimal_rl_20261116/submit_cell.sh
```

```bash
wsl -- ssh torch bash -s -- focus-smoke < research/hpc/minimal_rl_20261116/submit_cell.sh
```

```bash
wsl -- ssh torch bash -s -- focus 3 < research/hpc/minimal_rl_20261116/submit_cell.sh
```

The `3` is an example, as for the plain run.

**The deploy of 2026-11-19.** At 08:08 UTC on the user's word, once the plain run's links had left the queue,
the cell's source moved from e82026b to 2d4919d with a bundle of 95,891 bytes. The deploy found both suites with
the planned checksums, and the reference, the job's Python, the supervised pass's receipt and its evaluation in
place. It submitted nothing. Read on the cluster afterwards (08:13 UTC):

- the source is at 2d4919d with no tracked change; the supervised cell's source is still at 9c80d9a;
- the five scripts parse under the cluster's bash and hold no carriage return;
- the file of sequences has the local file's checksum (`499d3a8b...`) and loads under the job's Python (3.12.13)
  with its 2,093 cases; the driver and the evaluation import there with the cell's environment;
- both focused configs load with the focus and its examples on, and everything they name is in place but the
  training pass's folder and their own output folders, which their jobs write;
- no job of this user queued.

**The training pass, submitted 2026-11-19 at 19:17:48 UTC** on the user's word ("Yes, submit it"), with the
command above and its default of four parts: `mrl-train-pass` 23746858, 23746859, 23746861 and 23746863, each
two of the eight shards, and `mrl-train-pass-close` 23746865, which waits on all four (`afterany`) and writes
`out/train_pass/evaluation.json`. Read on the cluster just before (19:17 UTC): the source at 2d4919d with no
tracked change, no job of this user queued, the supervised pass's receipt and adapter in place, the file of
sequences listing its 2,093 cases, and no `out/train_pass` yet. All five jobs were pending when submitted.

**What the training pass is read for:** how many of the 2,093 cases the supervised model leaves unrepaired, and
in which classes. 1,966 of them were in the supervised pass's training rows and 127 in its validation rows, so it
may repair a larger share of them than the 80% of the development cases. If few are left, an iteration's 24 come
round again sooner, and the run teaches the same cases more often; the pass's count says how often.

**What the focused smoke has to show before a focused run:**

- the dry check's `focus` block: 2,093 cases with a sequence, and the split the pass gave;
- every replayed sequence repairs its case on the cluster (`replays that did not repair 0`): the sequences were
  built on Windows, and a replay that does not repair there is not kept;
- every truth-built action renders (`supervised_rows_unrenderable 0`, integrity `ok True`): the first time the
  target renderer meets the model's own tokenizer and chat template;
- the supervised loss is finite, and how many rows the per-row gradient cap of 1.0 clips;
- the update's peak memory with the example rows, and one whole pass through the phases.

**Checked locally.** The driver's tests (25) cover the draw, the refusals, a whole focused iteration with
stand-ins for the model (examples kept only where no sample repairs, their rows and weights, the update's counts
and integrity), a weight other than 1.0 scaling every example row, an action that does not render, a replay that
does not repair, a preempted update with examples resuming to the same weights, the dry check, and the rendering
of an action as the assistant turn of a supervised row (with a stand-in for the chat template that, like the
12B's, adds the empty thought channel to a generation prompt only). 17 hand-made faults in the new driver code
each fail one of them; the one that ignored the configured weight did so only once the weight test was added.
The target renderer was also run once against a real chat template of the family (the Gemma 4 E2B tokenizer
cached on the laptop): all six kinds of action rendered and read back as themselves, none refused. The cell's
tests run the training-pass stage and the focused stage against stubs: a part runs its two shards (at once on a
large card, one after the other on a smaller one), a failed shard fails its job, the closing job writes the
evaluation and a later one finds it, and the focused jobs refuse to start without the supervised pass's receipt
or without the training pass.

**Not checked.** It is deployed and its training pass is submitted, but none of it has finished on the cluster
or met the model. The target renderer has not met the 12B's
own processor: that is first exercised by the focused smoke. The replay of the 2,093 sequences on Linux is
untested. Whether teaching the truth-built
sequence on failed training cases repairs more of the 49 development cases is what the run is for.

## What the smoke showed that the run should know

Both are from six and eight cases at a checkpoint in the middle of the pass. They are things to look at in the
run's first iteration, not conclusions.

1. **The policy samples one episode most of the time.** With four samples a case and episode probabilities like
   the smoke's, about 3.5 of 6 cases would hold two different episodes at all (my arithmetic: one less the fourth
   power of each probability), and a different episode is not yet a different outcome: the smoke's one such case
   had none. An update learns a repair only from cases with a repaired and an unrepaired sample. The count is
   `groups_with_repaired_and_unrepaired_samples` in the rollout's summary, printed by `status_cell.sh`. The
   supervised pass's loss was still falling when the checkpoint was taken, so the finished model may sample less
   widely than this one. Samples per case and the sampling temperature are the settings that bear on it.
2. **An update with almost no gradient changed 3 of 8 greedy episodes.** The first step of a new AdamW is a sign
   step: it moves every trained weight by the learning rate whatever the gradient's size (`build_optimizer` in
   `psse_env/rl/update.py` says so), so a run's first update is as large with 9 rows of noise as with 1,000 rows
   of signal. That this is what changed the three episodes is my reading: I have not separated it from
   run-to-run variation of the evaluation itself (the two evaluations ran on the same card with the same split of
   cases, and five of eight episodes repeated exactly). If the reading is right, part of the first iteration's
   change in cases solved is this step's size, which the learning rate sets; from the second update on the
   carried optimizer scales a weight's step by how steady its gradient's sign has been.

## What the smoke has to show before a run

The smoke starts from a checkpoint early in the pass, so what it solves says nothing about the supervised model.
It is read for what only the model can show:

- every sampled step carries its token probabilities (`rows without a record 0`), and the share of steps whose
  tokens are all above 0.99: a policy that peaked samples one episode several times and gives the update nothing;
- every stored step renders back to its sampled prompt under the model's own tokenizer: no prompt length or digest
  mismatch, no cut prompt, integrity `ok True`;
- the gap between the sampler's log-probabilities and the update's first pass, overall and on the uncertain tokens
  (the batch is refused above a mean of 0.2);
- the update's peak memory on rows of up to 15,000 tokens with the reference adapter loaded (about 40 GB is my
  expectation), and the two rollout processes' peak together;
- the seconds a sampled episode and an update row take, which replace the estimates above;
- one whole pass through the phases: baseline, rollout, update, evaluation, the manifest at iteration 1.

`status_cell.sh` prints each of these per run and iteration.

## How to read the run

**The number a run is read by is cases solved of the 247 a truth-built sequence repairs** (the user's rule of
2026-11-18): the supervised model starts at 198 of 247, the rule expert is at 184, and 49 are left. The 53
cases neither reference repairs stay in the evaluation but not in that number.

Per iteration `status_cell.sh` prints the rollout (cases with both a repaired and an unrepaired sample are the
ones an update can learn a repair from), the update (rows, losses, the first-step gap, integrity) and the
evaluation: cases solved, how many of the 247, the split by reference (the 184 cases the rule expert's recorded episodes solve, the 63
only the truth-built sequence repairs, the 53 neither repairs), and the cases gained and lost against the
baseline. In the earlier cells one greedy pass over these cases moved by a few cases between card types, so a gain
is read from several iterations and from which cases moved, not from one count.

## Checked locally

`python -m pytest research/hpc/minimal_rl_20261116/test_cell.py` (24 tests, the focused run's among them): every
script parses and holds no
carriage return; scripts name only this cell's work directory and files that exist; nothing writes to the
supervised cell; the run's config is the driver's example on the cluster's paths and the smoke differs in size and
start alone; the episode and the limits are the ones the supervised cell evaluates under, so its evaluation can
stand as the baseline; the smoke takes the newest checkpoint that was written whole; the link count; the deploy
script submits nothing; and the status reader reads a run the driver wrote (with the driver tests' stand-ins for
the model).

Three of them run the stage's own shell, on a copy of the cell under a temporary work directory with a stub in
the driver's place: the smoke copies a whole checkpoint and runs every phase once (two shards and a closing call
for the baseline, the rollout and the evaluation, the update alone), a later job finds the target reached and
runs nothing, a higher target goes on from the manifest, an update that fails its checks stops that job and every
later one, and the run refuses to start without the supervised pass's receipt.

The driver's own tests: `python -m pytest research/minimal_env/test_rl_train.py` (25 tests).

Both suites on the cluster have the checksums the RL series planned, and the local files the demonstrations were
built from have the same ones (training `781c3d91...`, development `04e2ad09...`; read 2026-11-16). The deploy
checks them again.

## Not checked

The smoke ran every phase once on an H200. The run's first iteration ran them at full size on RTX PRO 6000
cards: the `run` branch of `submit_cell.sh`, the supervised cell's evaluation taken as the baseline, a 629-row
update, two later links that found the target reached, and a preemption in the evaluation that the requeued job
went on from. Not exercised on the cluster:

- a preemption in the rollout or the update (the update's checkpoint resume has run in the driver's tests only);
- a second iteration: the KL term away from zero, the optimizer carried from one update to the next, the draw of
  new cases;
- a prompt over 12,700 tokens (the run's longest prompt is not read);
- anything of the focused run ("The focused run", its "Not checked").
