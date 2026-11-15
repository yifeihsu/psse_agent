# The supervised pass of the minimal environment (prepared 2026-11-15)

Trains an adapter on the truth-built demonstrations executed through `psse_env/minimal` and evaluates it on the 300
development cases in that environment. Background and numbers: `docs/minimal_env_20261115.md`; the environment's
contract: `psse_env/minimal/README.md`.

**State: prepared only.** Not deployed, nothing submitted, and no script of this cell has run on the cluster. Each
of deploy, smoke and submission needs the user's word.

## What it runs

| Job | Stage | What |
| --- | --- | --- |
| `msf-row-gate` | `stage_row_gate.sbatch` (CPU) | every row rendered by the model's chat template with its own six tools and tokenized; its result is kept for the pass's restarts |
| `msf-train-START` | `stage_train.sbatch` | the pass: the trainer and every trainer setting of the earlier cloning passes (lr 3e-5, 1 x 4, LoRA 16, 4-bit, seed 20261107), entered through `research/minimal_env/train_sft.py` |
| `msf-eval-dev300-START` | `stage_eval.sbatch` | the pass's adapter, greedy, bf16, 16 actions, on the 300 development cases (`research/minimal_env/evaluate.py`) |
| `msf-smoke-START`, `msf-eval-smoke` | the same stages | two optimizer steps on about 200 rows; R1 on eight cases, read for prompt length, memory and refusals only |

Rows: `output/minimal_env_20261115/selection_ieee14` (14,823 training rows, 256 validation rows, 2,398 and 152
cases). 3,706 optimizer steps.

## The one decision in the cell: where the pass starts

`START` is `r1` or `base` (second word of every `submit_cell.sh` command; default `r1`). The two keep their output
apart (`out/pass_r1`, `out/pass_base`), so both can be run.

- `r1`: the adapter every earlier cloning pass started from. Same start as S0 and the open-rules clone; its weights
  were tuned on this domain, but on the old prompt and the old tools.
- `base`: a new LoRA on the base model. Nothing inherited from the rule expert's demonstrations, which is the
  cleaner test of "new workflow plus truth-built data"; the risk is that 3,706 steps at this learning rate are too
  few from scratch.

The default is `r1` because it is what every earlier pass did. It is the user's choice.

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

## What the smoke has to show before the pass

- The row gate passes on the smoke rows: these rows have never met the tokenizer, and the gate reads each tool call
  back from the rendering (`undo` with no arguments is the form no earlier row had).
- Two optimizer steps run and a resume would find a checkpoint.
- The evaluation smoke writes eight episode files and `evaluation.json`, with no `prompt_limit` outcome.

## How to read the evaluation

`status_cell.sh` prints, per evaluated adapter: cases solved (object level and strict audit), solved and declared
`resolved`, false `resolved`, damaged episodes, refused calls by code, and the split by reference
(`dev300_reference.json`): the 184 cases the rule expert's recorded episodes solve, the 63 only the truth-built
sequence repairs, the 53 neither repairs. For orientation in the old environment with the rules lifted: S0 186
(180 / 6 / 0), the clone 163 (143 / 20 / 0).

## Not checked

- No cell test file (`test_cell.py`) was written, unlike the other cells. Checked by hand on 2026-11-15: every script
  parses (`bash -n`), no file holds a carriage return, the rows manifest accepts the real selection (14,823 and 256
  rows, checksums as the selection report), lint is clean.
- `research/minimal_env/evaluate.py`, `row_gate.py` and the row-gate cache in `train_sft.py` have no unit tests yet.
- Nothing here has run with the model: the trainer entry, the row gate and the model policy are first exercised by
  the smoke.
