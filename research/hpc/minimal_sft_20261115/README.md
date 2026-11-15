# The supervised pass of the minimal environment (prepared 2026-11-15)

Trains an adapter on the truth-built demonstrations executed through `psse_env/minimal` and evaluates it on the 300
development cases in that environment. Background and numbers: `docs/minimal_env_20261115.md`; the environment's
contract: `psse_env/minimal/README.md`.

**State: deployed, nothing submitted.** Deployed on the user's word on 2026-11-15 (23:16 to 23:29 UTC) at commit
ea5e9a2: `/scratch/yx3882/minimal_sft_20261115/source` is at that commit, and the rows, their manifest and the
selection report are under `data/rows`. No job of this cell has been submitted, and no script of it has run with the
model. The smoke and the submission each need the user's word.

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
rank 16, 4-bit, seed 20261107, save every 16 steps, validate every 128.

One difference from BC0 is kept on purpose: BC0 chose its checkpoint by validation loss, and this pass takes its last
step, as the cloning passes did. The rows are seen once, and validation loss was a poor guide the last time (the
open-rules clone reached 0.019 and lost 23 cases).

## Estimates (mine, not measured)

Prompts are about 9,900 tokens with their tools at the median, against about 5,800 for the open-rules pass, which
took 25 to 27 s per step. At 40 to 60 s per step the pass is 41 to 62 GPU-hours; `submit_cell.sh` counts 60 s and
submits 7 links of 12 hours. The evaluation should take 2 to 4 hours. The longest row needs roughly 40 GB at
training time by the earlier rule of thumb.

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
  back from the template's rendering. The parser reads all six actions in the model's wire form, `undo` with no
  arguments included (`research/minimal_env/test_minimal_scripts.py`); what is untested is the template's own
  rendering of them.
- Two optimizer steps run from the base model at 1e-4 with a finite loss, and a resume would find a checkpoint.
- The evaluation smoke writes eight episode files and `evaluation.json`, with no `prompt_limit` outcome.

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

Nothing here has run with the model or its tokenizer: the trainer entry, the row gate and the model policy are first
exercised by the smoke.
