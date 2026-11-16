# RL in the minimal environment (deployed 2026-11-16)

Runs `research/minimal_env/rl_train.py` on the cluster: sampled episodes of `psse_env/minimal`, the outcome reward,
one on-policy update per iteration with a KL penalty to the supervised model, and the greedy evaluation on the 300
development cases. What an iteration does and why its settings are what they are:
`docs/minimal_env_20261115.md`, section 12.

**State: deployed on 2026-11-16 at 15:44 UTC, source at e82026b. Nothing is submitted and no job has run.** The
smoke and the run each need the user's word. The run also needs the supervised pass to finish
(`research/hpc/minimal_sft_20261115/README.md`); the smoke does not.

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

## Estimates (mine; nothing here is measured)

Per iteration of 64 cases by 4 samples, from the supervised cell's smoke rates: the rollout 1 to 2 hours with two
processes on a card, the update 4 to 7 hours, the evaluation 1.5 to 3 hours: roughly 8 to 14 GPU-hours. The
update's rate in bf16 is the least certain part. `submit_cell.sh` counts 14 hours an iteration and 11 hours of
each 12-hour link, plus a spare: 3 links for one iteration, 4 for two, 5 for three.

The smoke: about an hour (four model loads, 8 and 8 evaluated cases, 12 sampled episodes, an update of about 60
rows).

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

No job of this cell has run. The stage's shell has run only against the stub. The deploy script has run once
(above); the submit script has only been parsed; the status script has run on the cluster on the empty cell, and
its reader on a stand-in run locally. The driver's `--check` has not run on the cluster: the login node's 3 GB
cannot hold the suites, so the smoke's first step is the first time. The driver has not met the model. The smoke
is the first time any of it meets the model, the tokenizer or Slurm.
