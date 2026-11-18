# RL in the minimal environment (deployed 2026-11-16)

Runs `research/minimal_env/rl_train.py` on the cluster: sampled episodes of `psse_env/minimal`, the outcome reward,
one on-policy update per iteration with a KL penalty to the supervised model, and the greedy evaluation on the 300
development cases. What an iteration does and why its settings are what they are:
`docs/minimal_env_20261115.md`, section 12.

**State: the smoke passed (`mrl-smoke` 23435096, 2026-11-17, 22 minutes on an H200, source e82026b): every phase
ran once and every check of the update held. The run is not submitted.** The run needs the user's word, and the
supervised pass to finish (`research/hpc/minimal_sft_20261115/README.md`), which it did on 2026-11-18 at 07:54
UTC. Its evaluation is read: 198 of 300 (rule expert 184, truth-built sequences 247). That evaluation was made
under this run's settings (compared with its recorded settings under the cell's environment), so the run takes
it as its baseline. Two things the smoke
showed about the run itself are under "What the smoke showed that the run should know".

## What it runs

| Job | What |
| --- | --- |
| `mrl-smoke` | one iteration on 6 training cases by 2 samples and 8 development cases, baseline included; 4-hour limit |
| `mrl-run`, a chain | the run, to the iteration count named at submission; each link goes on from the run's manifest |

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

Commits on the branch after e82026b are records alone (the READMEs and the notes): the deployed source is not
behind in code.

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

Per iteration `status_cell.sh` prints the rollout (cases with both a repaired and an unrepaired sample are the
ones an update can learn a repair from), the update (rows, losses, the first-step gap, integrity) and the
evaluation: cases solved, the split by reference (the 184 cases the rule expert's recorded episodes solve, the 63
only the truth-built sequence repairs, the 53 neither repairs), and the cases gained and lost against the
baseline. In the earlier cells one greedy pass over these cases moved by a few cases between card types, so a gain
is read from several iterations and from which cases moved, not from one count.

## Checked locally

`python -m pytest research/hpc/minimal_rl_20261116/test_cell.py` (17 tests): every script parses and holds no
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

The driver's own tests: `python -m pytest research/minimal_env/test_rl_train.py` (14 tests).

Both suites on the cluster have the checksums the RL series planned, and the local files the demonstrations were
built from have the same ones (training `781c3d91...`, development `04e2ad09...`; read 2026-11-16). The deploy
checks them again.

## Not checked

The smoke has run every phase once with the model, on one H200. Not exercised on the cluster:

- the `run` branch of `submit_cell.sh`, a chain's hand-over from link to link, and a preempted job's resume (the
  resume has run locally only: the stage's shell against the stub, the update's checkpoint in the driver's tests);
- taking the supervised cell's evaluation as the baseline (`baseline_from`): the smoke ran its own;
- a second iteration: the KL term away from zero, the optimizer carried from one update to the next, the draw of
  new cases;
- an update of more than 27 rows, a prompt over 12,700 tokens, and any card but an H200 (two processes need
  63.4 GB together; the stage starts two from 70 GB of card memory).
