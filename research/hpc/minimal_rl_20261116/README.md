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

**The focused run is deployed (2026-11-19, source 2d4919d), and its training pass ran on 2026-11-19 (submitted
19:17 UTC on the user's word, evaluation written 22:10 UTC; `mrl-train-pass` 23746858, 23746859, 23746861,
23746863; `mrl-train-pass-close` 23746865): the supervised model leaves 502 of the 2,093 training cases a
truth-built sequence repairs unrepaired (24.0%, against 19.8% of the development cases). The focused smoke
passed on 2026-11-20 (`mrl-focus-smoke` 23799236, submitted 03:35 UTC on the user's word, completed 05:01 UTC
after one preemption, 39 minutes of card time on two H200s): every item on its list held, 6 of 6 replayed
sequences repaired their cases and 25 of 25 example actions rendered. The focused run was submitted on
2026-11-20 at 06:24 UTC on the user's word, to three iterations (`mrl-focus`, five links from 23814465): "The
focused run" below. Its first iteration was read on 2026-11-20 at 17:05 UTC: 199 of the 247 (gained 5, lost 4);
11 of its 24 failed cases had a repaired sample. Its second was read on 2026-11-21 at 02:06 UTC: 200 of the 247
(gained 7, lost 5 against the supervised model; sign test p = 0.77); the third iteration is running.** It draws from the
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

**The training pass, finished 2026-11-19 at 22:10:53 UTC, read 2026-11-20.** Every part was preempted and
requeued at least once (23746858 and 23746859 twice; their last instances ran on H200s, every other instance on
RTX PRO 6000s), and every one ended COMPLETED with no error in any shard's log. The closing job found all 2,093
episodes and wrote `out/train_pass/evaluation.json`: the supervised pass's adapter, greedy, bf16, 16 actions,
two table versions, input limit 32,768 tokens, output limit 256. About 9.1 GPU-hours, 2 h 50 min from the first
start to the evaluation.

- The supervised model repairs 1,591 of the 2,093 cases (76.0%) and leaves 502 (24.0%); of the development
  cases a truth-built sequence repairs it leaves 49 of 247 (19.8%). Of the 1,966 cases whose rows it trained on
  it leaves 469 (23.9%), of the 127 whose rows were held out 33 (26.0%): having trained on a case's sequence
  has not made the case repaired.
- Left unrepaired by class (of the class's cases): cb2_dangling 67 of 120, cb2_mixed 50 of 90, cb2_split 46 of
  90, control_cb 2 of 100, control_meter 11 of 100, control_rx 3 of 93, meter_on_rx 20 of 130, meter_rx2 70 of
  201, rx2_adjacent 35 of 177, rx2_distant 43 of 201, rx_cb_distant 62 of 273, rx_cb_meter 60 of 254,
  rx_cb_same_line 0 of 150, rx_cb_shared_bus 33 of 114. The two-breaker classes hold 163 of the 502.
- How the 502 end:

| Ending | Cases |
| --- | --- |
| declared unresolved, the final estimate fails | 217 |
| declared resolved, the final estimate fails | 93 |
| the budget ran out, the final estimate fails | 18 |
| declared resolved, the final estimate passes, a healthy object damaged | 55 |
| declared resolved, the final estimate passes, nothing damaged, an error left | 119 |

- 435 of the 502 repaired part of the case first. 251 of them had a refused call, against 28 of the 1,591
  repaired cases. The 643 refused calls: `correct_parameters` / `tool_failure` 413, `correct_topology` /
  `unknown_object` 216, `correct_topology` / `no_change` 13, `correct_topology` / `tool_failure` 1; 20 episodes
  with five or more hold 272 of them. (On the development cases: 58 and 30 of the first two, 88 in 300 cases.)
- Declared against the outcome over all 2,093: resolved 1,857 (1,590 of them repaired, 267 not), unresolved 218
  (217 not repaired), nothing declared 18; a healthy object damaged in 160 episodes.

What this means for the focused run: its draw has 502 unrepaired and 1,591 repaired cases, so at 24 unrepaired
cases an iteration each one comes round about once every 21 iterations. In 328 of the 502 the final estimate
fails, so the numbers the model reads show the case is not done; in the other 174 they pass.

**The focused smoke, submitted 2026-11-20 at 03:35:42 UTC** on the user's word ("Yes, smoke"):
`mrl-focus-smoke` 23799236, 4-hour limit, one iteration of `rl_focus_smoke.json` (6 unrepaired and 2 repaired
training cases by 2 samples, 8 development cases, the baseline included). Read on the cluster just before
(03:35 UTC): the source at 2d4919d with no tracked change, no job queued, the training pass's evaluation and the
supervised pass's receipt in place, and no `out/focus_smoke` yet.

**The focused smoke, completed 2026-11-20 at 05:01:59 UTC, read at 05:14 UTC.** Two instances, both on H200s.
The first (gh125) started at 04:05, 29 minutes after submission, ran the dry check, the baseline, the rollout and
the update, and was preempted at 04:36:51, 47 seconds into the evaluation, before it had written an episode. The
requeued job (gh103) started at 04:54:58, went on from the manifest with the evaluation, and ended COMPLETED at
05:01:59 with the last line `mrl-focus_smoke-complete`. 38 min 43 s of card time; no traceback in the job's log
or in any of its six shard logs. The manifest stands at iteration 1, phase rollout.

Against the list below ("What the focused smoke has to show before a focused run"):

| Item | Result |
| --- | --- |
| The dry check's `focus` block | ok, no problems, in both instances: 2,093 cases with a sequence, 502 left unrepaired and 1,591 repaired by the starting adapter (the training pass's split), 6 and 2 drawn, examples on at weight 1.0 |
| Every replayed sequence repairs its case | 6 of 6 (`replays that did not repair 0`, none without a sequence; 14.3 s for the six) |
| Every truth-built action renders | 25 of 25 rows (`supervised_rows_unrenderable 0`, `targets_refused 0`); integrity ok, no failure, no warning |
| The supervised loss, and the per-row cap | 0.092, finite. The 104 rows' gradient norms: median 0.0008, 95th percentile 0.34, largest 8.57; the cap of 1.0 clipped 2; none dropped as non-finite |
| The update's peak memory with the examples | 39,289 MiB, one process (the first smoke's update about 36,500 MiB); two processes 66,583 MiB (baseline), 63,587 MiB (rollout), 66,685 MiB (evaluation) |
| One whole pass through the phases | baseline 5 min 5 s, rollout 5 min 43 s (the six replays included), update 18 min 48 s, evaluation 5 min 37 s after the requeue; loading included |

- **The update.** 104 rows: 75 sampled rows of the 7 cases whose two samples differ in reward, 4 preservation
  rows of the one that does not, 25 example rows of the six examples (390 tokens, weights 1.105 to 1.785, mean
  1.47). 1,021 s, 9.8 s a row (the first smoke 9.2). Policy loss -0.0020, KL 0 (the reference is the start),
  gradient norm 1.87; the global clip of 1.0 does not change this step, the first of a new AdamW being a sign
  step. The first pass against the sampler: mean |log ratio| 0.0014 over 1,091 sampled tokens, 0.014 on the 115
  uncertain ones, largest 0.176, none beyond the threshold. Each row once, one optimizer step; the optimizer is
  saved for the next iteration.
- **The rollout.** The six cases the supervised model leaves unrepaired (cb2_dangling, control_meter,
  meter_rx2, rx_cb_distant, rx2_distant twice) were repaired in none of their 12 samples; the two it repairs
  (control_cb, rx2_distant) in all 4. 7 of the 8 cases sampled two different episodes; 35% of the 83 sampled
  steps had every token above 0.99 (the first smoke 53%, the plain run 56%). So no case had a repaired and an
  unrepaired sample, and each of the six got its example. Refused calls: 9 tool failures, 1 no change.
- **Where the samples went wrong**, against the example of the same case (the actions with their arguments):

| Case (class) | The truth-built example | The two samples |
| --- | --- | --- |
| `r0_0a67a3746f13` (cb2_dangling) | opens CB_4_N6_B2, inspects, opens CB_9_B1_N1 | both open CB_9_B1_N1, then fit a line (3; 6, refused) instead of looking for the second breaker |
| `r0_4576d58547ba` (rx2_distant) | fits lines 1 and 11 | both fit line 2 first; then line 11 and a meter repair, declared resolved, or line 18 (refused) and meter repairs |
| `r0_5445d8be4a0b` (rx_cb_distant) | opens CB_Y1014_10N1_10N2, fits line 4 | both fit line 4 and never open that breaker (one tries CB_13R2_13R3, refused as no change); each sends line 18 three times, refused |
| `r0_5508b87ff3db` (control_meter) | repairs meters 14 and 43 | both repair meter 43 without 14 and declare unresolved |
| `r0_c6fa8de52cb0` (rx2_distant) | fits lines 5 and 16 | both open CB_2R2_2R3, a healthy breaker, then a line-7 fit or meter repairs; neither touches line 5 or 16 |
| `r0_cb69ce62b9a2` (meter_rx2) | fits lines 7 and 6, repairs meter 67 | both fit line 3 instead of 7 and repair meter 67 (one also fits line 6); both declare resolved |

  In five of the six a sample made at least one of the example's corrections, and what was missing is one
  object: a second breaker or meter not looked for, a breaker not found, or the wrong one of two lines. In the
  sixth the samples went to a breaker on a case of two line errors. The supervised loss of 0.092 is the example
  rows' mean token log-loss weighted by their weights, about 0.06 a token unweighted. My reading, not measured
  (the per-token losses are not recorded): the starting model already gives most tokens of the truth-built
  actions a high probability, and what the examples teach is in the few tokens that name the object.
- **The development cases** are the suite's first eight, all cb2_dangling (four the rule expert's episodes
  solve, four only the truth-built sequence repairs): 4 of 8 solved before the update and 4 after, none gained,
  none lost. Five episodes keep their tool sequence; three changed: `r0_77b3ec119344` dropped a fit-and-undo
  detour (7 actions to 5, repaired both times), `r0_891df54d8682` declares unresolved instead of resolved (not
  repaired either way), `r0_72c33dbda5f8` makes one parameter fit fewer (not repaired either way). Two greedy
  passes of the starting model differ as well: this baseline against the supervised cell's evaluation of the same
  eight cases (same adapter and settings, another job) has the same tool sequence in 6 of 8 and the same outcome
  in 8 of 8, and the two that differ are `r0_891df54d8682` and `r0_72c33dbda5f8`. Eight cases of one class are
  not a reading of the update.

**A full focused iteration, my arithmetic from the measured rates.** The plain run's iteration on RTX PRO 6000
cards: rollout 1 h 12 min for 256 episodes, update 2.28 h for 629 rows (13.0 s a row), evaluation about 1 h 20
min. A focused iteration samples the same 256 episodes, but its update will be larger: in the smoke 7 of 8 cases
differed in reward (the plain run's first iteration 23 of 64), since the samples of a case the model fails seldom
share a reward. At that share, about 28 of 32 cases by 8 samples by 5.2 actions is about 1,160 sampled rows; with
about 100 example rows, about 1,250 rows, 4.5 hours at 13.0 s a row (3.4 at the smoke's 9.8 s on an H200). So
about 7 to 7.5 hours an iteration on RTX PRO 6000 cards, against the 14 hours `submit_cell.sh` counts per
iteration when it sets the links (`focus 3`: 5 links of 12 hours, each leaving at once when the target is
reached).

**The focused run, submitted 2026-11-20 at 06:24:05 UTC** on the user's word ("Yes, focus 3"), with
`submit_cell.sh focus 3` from the source at 2d4919d: three iterations, five links, `mrl-focus` 23814465,
23814468, 23814470, 23814471 and 23814472, each after the one before it (`afterany`). Read on the cluster just
before (06:23 UTC): the source at 2d4919d with no tracked change; `rl_focus.json`, `stage_rl.sbatch`,
`cell.env`, `submit_cell.sh` and the file of sequences with the local files' checksums (`rl_focus.json`
`17037a4f...`); no job of this user queued; everything the config names in place (the supervised pass's adapter
and receipt, both suites, the reference, the supervised cell's evaluation as the baseline, the training pass's
evaluation); no `out/focus` yet. All five links were pending when submitted.

What it is read for, per iteration (`status_cell.sh` prints each):

- the evaluation on the 300 development cases: solved of the 247 (the supervised model 198, the plain run's
  iteration 202), the cases gained and lost against the 198, and which;
- the rollout: of the 24 unrepaired cases, how many have a repaired sample among their eight (they teach by
  contrast) and how many get an example instead;
- the update: its rows (about 1,250 by the arithmetic above), seconds a row, memory and integrity; from the second
  iteration the KL term away from zero and the optimizer carried, neither run on the cluster before.

**The first iteration, written 2026-11-20 by 16:59 UTC, read 17:05 UTC** (link 23814465 on gr105, an RTX PRO 6000,
from about 09:34 UTC; the update phase alone 16,291 s, 4.5 h):

| | |
| --- | --- |
| Development, solved of the 247 | 199 (the supervised model 198, the plain run's iteration 202): gained 5, lost 4 |
| ... of the 184 the rule expert solves / of the 63 only the truth-built sequence repairs | 163 (161) / 36 (37) |
| false resolved, declared unresolved, out of actions, episodes with damage (of 300) | 71, 26, 4, 25 (72, 27, 3, 27) |
| Rollout: the 24 unrepaired cases | 11 with a repaired and an unrepaired sample, 13 with none; 28 of 192 episodes repaired |
| Rollout: the 8 repaired cases | 64 of 64 episodes repaired |
| Groups with a reward spread / steps with every token above 0.99 | 26 of 32 / 36% of 1,305 |
| Refused calls in the rollout | 298 (tool failure 178, unknown object 116, no change 3, unparseable 1) |
| Truth-built examples | 13, for the 13 cases without a repaired sample; 56 rows, weights 1.01 to 1.76, supervised loss 0.104 |
| Update | one step on 1,186 rows (1,105 sampled, 25 preservation, 56 example); policy loss -0.0013, KL 0, gradient norm 3.91 |
| Against the sampler | mean abs log ratio 0.0014 (uncertain tokens 0.017), largest 0.66; integrity ok |

Gained: r0_7449ca0d5cd6 (meter_on_rx), r0_b37da0e8deca (cb2_dangling), r0_c35a1e4bfa40 (rx2_distant),
r0_e802f3b2ab39 (meter_rx2), r0_e8b622e42551 (cb2_split). Lost: r0_7b9e211f46d8 (rx2_adjacent), r0_891e3e9681b2
(meter_on_rx), r0_c19207e4273d (rx_cb_meter), r0_c83a69b4802b (cb2_mixed). 5 against 4 is no measurable change.
Sampling found a repair in 11 of the 24 failed cases, far more than the smoke's 0 of 6, so the run has contrast to
learn from; one step at 1e-5 moved the greedy result by one case. The run went on to its second iteration.

**The second iteration, evaluation written 2026-11-21 by 02:02 UTC, read 02:06 UTC.** Its rollout ran on gr105
(16:53 to 18:20 UTC); the update was preempted three times and resumed each time from its saved rows (on gr103,
gh101 and gr105; the last stint resumed at 1,088 of 1,194 rows), and the evaluation once (resumed on gr102).

| | |
| --- | --- |
| Development, solved of the 247 | 200 (the supervised model 198, the first iteration 199, the plain run's iteration 202): gained 7, lost 5 against the supervised model |
| ... of the 184 the rule expert solves / of the 63 only the truth-built sequence repairs | 165 (161) / 35 (37) |
| false resolved, declared unresolved, out of actions, episodes with damage (of 300) | 72, 25, 3, 31 (72, 27, 3, 27) |
| Rollout: the 24 unrepaired cases | 12 with a repaired and an unrepaired sample, 12 with none; 31 of 192 episodes repaired |
| Rollout: the 8 repaired cases | 60 of 64 episodes repaired (2 cases with an unrepaired sample) |
| Groups with a reward spread / steps with every token above 0.99 | 25 of 32 / 42% of 1,312 |
| Refused calls in the rollout | 246 (tool failure 178, unknown object 65, no change 3) |
| Truth-built examples | 12, for the 12 cases without a repaired sample; 57 rows, weights 1.03 to 1.82, supervised loss 0.106 |
| Update | one step on 1,194 rows (1,112 sampled, 25 preservation, 57 example); policy loss -0.0019, KL 0.0033, gradient norm 2.95; the optimizer carried from the first iteration (step 2) |
| Against the sampler | mean abs log ratio 0.0016 (uncertain tokens 0.018), largest 1.34; integrity ok |

Gained against the supervised model: r0_b37da0e8deca (cb2_dangling), r0_e802f3b2ab39 (meter_rx2), r0_e8b622e42551
(cb2_split), all three also gained by the first iteration, and r0_17b04fcf34bb (meter_rx2), r0_a04d7fb1791e
(rx2_adjacent), r0_b47550259680 (cb2_split), r0_bd2c9367bbe9 (cb2_dangling). Lost: r0_7b9e211f46d8 (rx2_adjacent),
r0_891e3e9681b2 (meter_on_rx), r0_c83a69b4802b (cb2_mixed), all three also lost by the first iteration, and
r0_5bae721d68ba (meter_rx2), r0_a7014fc5a4e1 (rx_cb_meter). The first iteration's r0_7449ca0d5cd6 and
r0_c35a1e4bfa40 were lost again, its r0_c19207e4273d recovered. By the stopping-state diagnostic's failure patterns
(`research/hpc/minimal_diag_20261120/README.md`), the 49 unrepaired cases now solved: second breaker left 4 of 17,
faulted line never fitted 2 of 10, wrong line fitted 1 of 9, breaker never corrected 0 of 7, the rest 0 of 6. The
gains are the cases where the supervised model was nearly right: of the 7 cases whose step on was within 1 nat of
the model's own action (at the stop or at the first wrong turn), 5 are solved; of the 18 within 1 to 3 nats, 2; of
the 24 further than 3 nats, none (that diagnostic's "Against the focused run's second evaluation").

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

**Not checked.** Its training pass and its smoke have run (above): the target renderer has met the 12B's own
processor (25 of 25 rows rendered), and 6 of the 2,093 sequences were replayed on Linux, all repairing their
cases; the other 2,087 have not been replayed there (a replay that does not repair is not kept, and the count
says so). A focused update at full size has run (1,186 rows, 4.5 h, integrity ok). The second iteration has
exercised the KL term away from zero (0.0033), the optimizer carried (step 2), a new draw, and three preemptions
inside the update, each resumed from its saved rows with integrity ok. Not exercised yet: a preemption inside the
rollout. Whether teaching the truth-built sequence on failed training cases repairs more of the 49 development
cases is what the run is for.

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
