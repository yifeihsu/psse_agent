# The supervised pass of the minimal environment (prepared 2026-11-15)

Trains an adapter on the truth-built demonstrations executed through `psse_env/minimal` and evaluates it on the 300
development cases in that environment. Background and numbers: `docs/minimal_env_20261115.md`; the environment's
contract: `psse_env/minimal/README.md`.

**State: the smoke ran; its training half failed and is fixed here, not yet on torch.** Deployed on the user's word
on 2026-11-15 (23:16 to 23:29 UTC) at commit ea5e9a2, and the smoke submitted on their word at 23:40 UTC. The
evaluation smoke passed. The training smoke ran out of card memory at its first step (below). The fix is committed
in this branch; torch still has ea5e9a2. A redeploy, the training smoke again and the submission of the pass each
need the user's word.

## The smoke of 2026-11-15

Both jobs started two minutes after submission on one node (RTX PRO 6000, 97,887 MiB).

**Evaluation smoke, `msf-eval-smoke` 23284025: completed in 13.5 minutes.** Eight episode files and
`evaluation.json`; no prompt refused for length; no step cut at the new-token limit. Prompts over 127 steps: median
14,644, longest 15,523 tokens. Two processes on the card peaked at 67.7 GB together, at 84% mean utilization.
Generation takes 3.5 s at 5,300 prompt tokens and 12 to 13 s at 14,700; an episode that uses all 16 actions takes
about three minutes. R1 was trained on another prompt and other tools, so what it does says nothing about the
pass: it repaired 2 of 8, used all 16 actions in 6, and 88 of its 127 actions were `inspect`. Its 18 refused calls
are the tools' own: 15 breaker changes the operator model cannot carry (an unsupplied island, an equivalent
state), one breaker already in the named status, one breaker that does not exist, and one line fit. The line fit
was reproduced locally: the provider's multi-scan fit wants scans of `3 x buses + 4 x lines` values for the model
it is given, so it refuses a line fit in a model where a breaker change has split a bus the recorded scans do not
split. That is the fit of the old environment too, unchanged here.

**Training smoke, `msf-smoke-base` 23284023: failed after 2 min 53 s.** The row gate passed on its 232 rows, the
first time these rows met the model's chat template: every tool call read back from the rendering, no row cut,
all six tools kept in every row, 0.23 s a row. Lengths with the tools, by the model's own tokenizer: median 9,956,
longest 15,237 tokens (training rows; the smoke takes the longest first); a target is 13 tokens at the median and
at most 37. Then CUDA out of memory before the first optimizer step: "tried to allocate 14.88 GiB" with 80.55 GiB
already allocated. 14.88 GiB is the float32 logits of every position of the 15,237-token row (15,237 x 262,144
x 4 bytes); the model's own loss holds several such copies. No checkpoint and no loss were written.

This limit was known. The complete-evidence clone met it four hours earlier (job 23274632) and its fix,
`--supervised-logits-only`, was committed as 5726b52 on `claude/open-rules-20261112`, after this branch had left
that one. The cell was prepared without it. Fixed here: that commit is brought in (2f8660e), the training stage
passes the option, and the deploy refuses a commit without it. The option computes the same loss from the logits
of the target's span alone (`psse_env/sft/supervised_logits.py`); it has not yet run with the model in either
cell, so the training smoke has to run again, and `status_cell.sh` prints how the forward was called (`kept_calls`
must equal the calls with labels; `ignored_calls` or `refused_calls` above zero means the memory was not saved).

`submit_cell.sh smoke` submits the training smoke alone now: it finds the evaluation smoke's `evaluation.json`
and leaves it. The smoke rows and the gate's timing are on disk and are reused.

Still to be shown by the training smoke: a finite loss over two optimizer steps from the base model at 1e-4, a
checkpoint a restart would resume from, the seconds per step, and the card memory at the longest row.

Checked on torch after the deploy (one read-only call): the source is at ea5e9a2 with no tracked change; the seven
scripts parse and hold no carriage return; the four uploaded files have the local checksums (training view
`582349e2...`, 14,823 rows; validation view `e55689a6...`, 256 rows); the 25 Python files of `psse_env/minimal`,
`research/minimal_env` and this cell compile under the jobs' Python 3.12.13; R1, the development suite, the
reference and the base model's cache are where the cell reads them; no `msf-` job is queued.

## What it runs

| Job | Stage | What |
| --- | --- | --- |
| `msf-row-gate` | `stage_row_gate.sbatch` (CPU) | every row rendered by the model's chat template with its own six tools and tokenized; its result is kept for the pass's restarts |
| `msf-train-START` | `stage_train.sbatch` | the pass, entered through `research/minimal_env/train_sft.py` |
| `msf-eval-dev300-START` | `stage_eval.sbatch` | the pass's adapter, greedy, bf16, 16 actions, on the 300 development cases (`research/minimal_env/evaluate.py`) |
| `msf-smoke-START`, `msf-eval-smoke` | the same stages | two optimizer steps on about 200 rows; R1 on eight cases, read for prompt length, memory and refusals only |

Rows: `output/minimal_env_20261115/selection_ieee14` (14,823 training rows, 256 validation rows, 2,398 and 152
cases). 3,706 optimizer steps.

## Where the pass starts

`START` is `base` or `r1`: the second word of every `submit_cell.sh` command. The two keep their output apart
(`out/pass_base`, `out/pass_r1`), so both can be run.

**The default is `base`** (the user's choice of 2026-11-15): a new LoRA on the base model, with nothing inherited
from the rule expert's demonstrations. `r1` is the adapter every earlier cloning pass started from.

Two trainer settings follow the start, and nothing else does:

| | `base` | `r1` |
| --- | --- | --- |
| `--initial-adapter` | none | R1 (`/scratch/yx3882/compound_cloning_20261106/data/start_adapter`) |
| Learning rate | 1e-4 | 3e-5 |

The learning rates are the project's own for each kind of pass. BC0, the one earlier pass from the base model,
trained at 1e-4 (`research/hpc/full_pipeline_20260907/pipeline.env`); every pass that continued an adapter trained at
3e-5. Everything else is the cloning passes' setting, one for one: micro-batch 1 with 4 accumulation, one epoch, LoRA
rank 16, 4-bit, seed 20261107, save every 16 steps, validate every 128. One option is added to them,
`--supervised-logits-only`: the same loss, computed without a vocabulary row for every prompt position (see the
smoke above).

One difference from BC0 is kept on purpose: BC0 chose its checkpoint by validation loss, and this pass takes its last
step, as the cloning passes did. The rows are seen once, and validation loss was a poor guide the last time (the
open-rules clone reached 0.019 and lost 23 cases).

## Estimates (mine, not measured)

Prompts are 9,956 tokens with their tools at the median (the row gate's count on the smoke rows), against about
5,800 for the open-rules pass, which took 25 to 27 s per step. At 40 to 60 s per step the pass is 41 to 62
GPU-hours; `submit_cell.sh` counts 60 s and submits 7 links of 12 hours. The seconds per step are still unmeasured:
the training smoke stopped before its first step.

The evaluation, from the smoke's timing (my arithmetic): an adapter that stops after 4 to 7 actions, as the
demonstrations do, should take 1.5 to 3 hours for the 300 cases with two processes on a card; one that uses all 16
actions on every case would take about 7 hours with two processes and 14 with one. The chain is two links of 12
hours and resumes from the episode files. Two processes share a card from 70 GB: they peaked at 67.7 GB in
episodes that ran to the action limit (prompts up to 15,523 tokens; only two versions' tables are shown, so a
prompt stops growing there), which leaves about 14 GB on an 80 GB card.

## Order of work

```bash
bash research/hpc/minimal_sft_20261115/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch bash -s -- smoke < research/hpc/minimal_sft_20261115/submit_cell.sh
```

```bash
wsl -- ssh torch bash -s < research/hpc/minimal_sft_20261115/status_cell.sh
```

```bash
wsl -- ssh torch bash -s -- all < research/hpc/minimal_sft_20261115/submit_cell.sh
```

The deploy reads the rows from this checkout's `output/minimal_env_20261115/selection_ieee14`; they were built in the
worktree `dreamy-bassi-27849a`, so from another checkout name the folder with `SELECTION_DIR`.

## What the smoke has to show before the pass

- The row gate passes on the smoke rows. These rows have never met the tokenizer, and the gate reads each tool call
  back from the template's rendering. **Shown 2026-11-15.**
- Two optimizer steps run from the base model at 1e-4 with a finite loss, and a resume would find a checkpoint.
  **Not shown: out of memory before the first step; fixed here, to be run again.**
- The evaluation smoke writes eight episode files and `evaluation.json`, with no `prompt_limit` outcome.
  **Shown 2026-11-15.**

## How to read the evaluation

`status_cell.sh` prints, per evaluated adapter: cases solved (object level and strict audit), solved and declared
`resolved`, false `resolved`, damaged episodes, refused calls by code, and the split by reference
(`dev300_reference.json`): the 184 cases the rule expert's recorded episodes solve, the 63 only the truth-built
sequence repairs, the 53 neither repairs. For orientation in the old environment with the rules lifted: S0 186
(180 / 6 / 0), the clone 163 (143 / 20 / 0).

## Checked locally

`python -m pytest research/hpc/minimal_sft_20261115/test_cell.py` (17 tests): every script parses and holds no
carriage return; scripts name only this cell's work directory and files that exist; the trainer's options are the
cloning pass's one for one but for the start; each start resolves to its adapter and learning rate; the link count;
the deploy script submits nothing; the manifest refuses rows that are not the selection's. The manifest accepts the
real selection (14,823 and 256 rows, checksums as the selection report).

## Not checked

The trainer has not taken a step on these rows, and `--supervised-logits-only` has not run with the model: the
training smoke is where both are first exercised. The row gate, the model policy and the evaluation have run with
the model (the smoke above).
