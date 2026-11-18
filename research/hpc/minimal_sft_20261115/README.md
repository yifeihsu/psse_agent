# The supervised pass of the minimal environment (prepared 2026-11-15)

Trains an adapter on the truth-built demonstrations executed through `psse_env/minimal` and evaluates it on the 300
development cases in that environment. Background and numbers: `docs/minimal_env_20261115.md`; the environment's
contract: `psse_env/minimal/README.md`.

**State: the pass has finished (2026-11-18 07:54 UTC, step 3,706 of 3,706, adapter and receipt written); its
evaluation on the 300 development cases is queued and not read.** Deployed on the user's
word on 2026-11-15 (23:16 to 23:29 UTC) at commit ea5e9a2, and the smoke submitted on their word at 23:40 UTC. The
evaluation smoke passed. The training smoke ran out of card memory at its first step; the fix was redeployed on the
user's word on 2026-11-16 (00:51 to 00:59 UTC, torch at 9c80d9a) and the training smoke passed on its second run.
The row gate, the pass and its evaluation were then submitted on the user's word.

## The full run, submitted 2026-11-16

`submit_cell.sh all` at 03:34 UTC, source 9c80d9a, start `base`. Read on torch in one call before it: the source at
9c80d9a with no tracked change, 14,823 and 256 rows, no row gate result yet, `out/pass_base` empty, no `msf-` job
queued.

| Job | Ids | Starts after |
| --- | --- | --- |
| `msf-row-gate` (CPU, 3-hour limit) | 23308022 | nothing |
| `msf-train-base`, 7 links | 23308024 to 23308030 | the row gate ending well; then each link after the one before it |
| `msf-eval-dev300-base`, 2 links | 23308031, 23308032 | the last training link ending well; then the second after the first |

The first training link starts only if the row gate ends well, so a failed gate leaves both chains queued with a
dependency that cannot be met: read the gate's log and cancel them. The complete-evidence clone's training chain
(`cec-train`, another cell) had one card when this was submitted; the two share this user's card limit.

**Row gate, 23308022: passed in 5 min 44 s** (read 03:45 UTC). All 15,079 rows rendered through the model's chat
template with their six tools and tokenized; every tool call read back; no prompt and no target cut; no row without
a supervised token; no row changed. 0.021 s a row and 15.7 GB of memory at the peak (the smoke's 0.23 s a row was
mostly start-up, and my hour for this job was wrong by that factor). The cache is 2.8 GB and a training link loads
it in 83 s.

| Tokens with the tools | Median | 95% | 99% | Longest |
| --- | --- | --- | --- | --- |
| Training rows (14,823) | 9,896 | 12,558 | 14,826 | 15,237 |
| Validation rows (256) | 9,912 | 12,564 | 14,908 | 15,099 |
| Target, training rows | 13 | 31 | 37 | 37 |

**The pass, first link 23308024:** started at 03:42 UTC on an H200, 90 seconds after the gate ended. Read at
05:52 UTC, 2 h 10 min in: step 192 of 3,706, no preemption, no loss that is not finite.

- Validation at step 128 (256 rows, 6 minutes): loss 0.104, token accuracy 0.960. The smoke's 1.241 was after two
  steps.
- Training loss by block of 64 steps: 0.271, 0.131, 0.102; token accuracy 0.937, 0.958, 0.970.
- Gradient norms before clipping average 84 to 146 a block and reach about 1,080; the trainer's default clip
  applies, as in the smoke.
- 37 s a step between checkpoints (35 to 40; 62 where the validation fell), 22.2 GB of card memory, 90% mean
  utilization.

At that rate the 3,706 steps take 38 hours and the 28 validations 3 more: about 41 hours on this card (my
arithmetic), so the fourth link should finish the pass if the later links get a card of the same speed.

**First preemption, and the first resume** (read 06:38 UTC). The job was preempted at 06:05 UTC at step 214, after
2 h 23 min on the H200. Slurm requeued it; it started again 7 minutes later on an RTX PRO 6000, loaded the row
gate's cache in 84 s and resumed from `checkpoint-208`, the newest complete one, so six steps were run twice.
`checkpoint-224` was written at 06:28 UTC; the loss over steps 193 to 224 averages 0.098. This is the requeue and
resume the smoke could not exercise. The rate on the new card is not read yet (the first interval holds the
restart).

**Seven preemptions by 12:02 UTC, then a steady stretch** (read 15:53 UTC). Between 06:05 and 12:02 UTC the job
was preempted seven times, six of them on an H200, and each time it resumed from its newest complete checkpoint
(432, 480, 512 and 560 for the last four). In those six hours it went from step 208 to step 560: 60 s a step,
restarts included. Since 12:06 UTC it has run on an RTX PRO 6000 (gr102) without a preemption: step 816 of 3,706
at 15:47 UTC, 12 h 5 min after the start.

- Rate on the RTX PRO 6000: 40 to 52 s a step between checkpoints, 83 to 95 where a validation fell (a validation
  takes 592 s there against 361 to 366 s on an H200). From step 560 to 816: 52 s a step, the restart and two
  validations included.
- Training loss by block of 64 steps from step 193: 0.101, 0.095, 0.100, 0.095, 0.097, 0.108, 0.080, 0.071, 0.111,
  and 0.094 over steps 769 to 816. It has stayed near 0.10 since step 130.
- Validation loss at steps 128, 256, 384, 512, 640 and 768: 0.104, 0.083, 0.098, 0.074, 0.120, 0.072; token
  accuracy 0.960, 0.971, 0.973, 0.979, 0.971, 0.975. No trend after step 256.
- Gradient norms before clipping reached 10,413 (steps 321 to 384) and 13,030 (steps 577 to 640); no loss that is
  not finite in the 816 logged steps.
- 22.4 GB peak card memory, 91% mean utilization.
- Slurm's clock for the job starts over at each requeue (3 h 48 min elapsed at the read), so the first link's
  12-hour limit now falls at 00:05 UTC on 2026-11-17 if it is not preempted first.

The 2,890 steps left take about 42 hours at this card's rate (my arithmetic): the pass would end near 10:00 UTC on
2026-11-18, later with more preemptions. What the flat loss means for the cases solved is not known before the
evaluation.

**Step 1,648 after sixteen preemptions** (read 2026-11-17 05:09 UTC). `checkpoint-1648` was written at 04:41 UTC,
25 hours after the start: 55 s a step over the whole pass, restarts and validations included. Nine more
preemptions since the read above, on H200, A100 and RTX PRO 6000 cards; every one resumed from the newest
complete checkpoint, no traceback, no loss that is not finite in the 1,648 logged steps. The job is still the
first link, 23308024: no link has reached its 12-hour limit, because each requeue starts Slurm's clock again (the
limit now falls at 16:49 UTC on 2026-11-17).

- The loss rose and came back. Training loss by block of 64 steps: 0.119 and 0.102 (steps 833 to 960), then 0.157,
  0.159, 0.181 (steps 961 to 1,152), 0.119, and from step 1,217 on 0.058, 0.071, 0.078, 0.061, 0.068, 0.051, and
  0.046 over steps 1,601 to 1,648, the lowest so far.
- Validation loss at steps 896, 1,024, 1,152, 1,280, 1,408 and 1,536: 0.083, 0.156, 0.150, 0.090, 0.086, 0.071;
  token accuracy 0.976, 0.951, 0.957, 0.975, 0.977, 0.980. The validation rows show the same rise at steps 1,024
  and 1,152, so it was the model and not the order of the training rows. I have not looked for its cause.
- Gradient norms before clipping reached 4,310 in the block where the rise began (steps 961 to 1,024) and stayed
  under 1,700 after it.
- 22.5 GB peak card memory, 91% mean utilization.

The 2,058 steps left take about 31 hours at the pass's overall rate (my arithmetic): it would end near 12:00 UTC
on 2026-11-18.

**Step 3,472, and the first hand-over between links** (read 2026-11-18 05:19 UTC). `checkpoint-3472` was written
at 05:14 UTC, 49.5 hours after the start: 51 s a step over the whole pass.

- The first link, 23308024, was preempted 22 times in all. From 12:32 UTC on 2026-11-17 it ran 11 h 52 min on one
  H200 without a preemption, from step 2,096 to step 3,157 at 40 s a step, until Slurm ended it at its 12-hour
  limit at 00:32 UTC on 2026-11-18 (state `TIMEOUT`).
- The second link, 23308025, started 69 seconds later and resumed from `checkpoint-3152`, so five steps ran
  twice. This is the hand-over the chain exists for, exercised for the first time. The link was preempted twice
  in its first 80 minutes (on an RTX PRO 6000 at step 3,194, on an A100 during the validation at step 3,200) and
  has run on an H200 since 01:51 UTC at 35 to 40 s a step. The pass has now trained on H200, RTX PRO 6000 and
  A100 cards.
- Training loss by block of 64 steps: 0.05 to 0.06 over steps 1,601 to 2,048; 0.038 to 0.047 over steps 2,049 to
  2,432; 0.025 to 0.043 from step 2,497 on, 0.031 over the last whole block (steps 3,393 to 3,456).
- Validation loss at steps 1,664 to 3,456, every 128: 0.080, 0.058, 0.061, 0.060, 0.078, 0.052, 0.056, 0.058,
  0.046, 0.036, 0.044, 0.044, 0.042, 0.042, 0.046; token accuracy from 0.979 to 0.989. The lowest is 0.036 at step
  2,816; since step 2,944 it has stayed between 0.042 and 0.046.
- No traceback, no loss that is not finite in the 3,472 logged steps; one gradient norm of 7,113 before clipping
  (steps 3,009 to 3,072), otherwise under 1,200 since step 2,049.
- Checkpoints are pruned when a job starts, not while it trains: every checkpoint since 3,152 is on disk (21 of
  them beside the every-128th ones, about 17 GB) until the next start or the end of the pass.

The 234 steps left and one more validation take about 2.6 hours on this card (my arithmetic): the pass would end
near 08:00 UTC on 2026-11-18 if it is not preempted. The five training links still queued then each start and
leave at once, and the evaluation on the 300 cases follows them.

**The pass ended** (read 2026-11-18 07:58 UTC). The second link, 23308025, completed at 07:54 UTC after 6 h 4 min
on one H200, with no further preemption: step 3,706 of 3,706, one pass over the 14,823 rows, last line
`msf-train-full-base-complete`.

- From the first step to the last: 52 h 12 min of wall time, 24 preemptions, two links.
- The adapter is `out/pass_base/training/lora` (`adapter_model.safetensors`, 262,373,216 bytes, sha256
  `173f99df...`); the receipt `out/pass_base/training.done` was written at 07:54:31 UTC with
  `research_run_passed` true.
- The adapter was loaded again on a fresh base model and its one canary row gave the expected action
  (`correct_parameters`, line 3).
- Final validation, step 3,706: loss 0.0445, token accuracy 0.9885. The two before it: 0.0465 (step 3,456) and
  0.0441 (step 3,584). The lowest of the pass stays 0.036 at step 2,816; that checkpoint is on disk.
- Training loss by block of 64 steps over the end: 0.032, 0.030, 0.034, and 0.038 over steps 3,649 to 3,706. The
  learning rate fell linearly to zero over the pass (5e-8 at step 3,705).
- Checkpoints left: every 128th, and 3,680, 3,696 and 3,706.

The five later training links (23308026 to 23308030) and the two evaluation links were still queued at the read;
the evaluation had not started.

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
that one. The cell was prepared without it. Fixed: that commit is brought in (2f8660e), the training stage passes
the option, and the deploy refuses a commit without it (9c80d9a). The option computes the same loss from the
logits of the target's span alone (`psse_env/sft/supervised_logits.py`), and `status_cell.sh` prints how the
forward was called (`ignored_calls` or `refused_calls` above zero would mean the memory was not saved).

**Training smoke again, `msf-smoke-base` 23290175 (2026-11-16, source 9c80d9a): completed in 6 min 16 s on an
H200.** It started 24 minutes after submission. `submit_cell.sh smoke` submitted it alone (the evaluation smoke's
`evaluation.json` was found and left); the smoke rows were reused and the row gate's result was loaded from its
cache.

- Two optimizer steps from the base model at 1e-4: loss 2.104 and 1.944, token accuracy 0.80 and 0.85, gradient
  norms 35.5 and 201.6 before clipping (the cell sets no clipping of its own; the trainer's default applies).
  Validation after the second step, 32 rows: loss 1.241, token accuracy 0.87. These are the first two steps of a
  new adapter, not a result. For scale: the complete-evidence clone's smoke, which continues R1, began at 0.49.
- `checkpoint-1` and `checkpoint-2` were saved, a resume would start from `checkpoint-2`, the receipt is written,
  `research_run.json` passed, and the reloaded adapter produced a parseable tool call (not the target's, after two
  steps).
- The span-only loss was used on every forward with labels: 41 kept (one optimizer smoke, eight training rows,
  32 validation rows), none ignored, none refused; at most 34 logits kept of 15,237 positions.
- Memory, sampled each minute: 34.7 GB at the highest sample, 21.9 GB during the two steps. The first run had
  80.55 GiB allocated when it failed.
- Rate: 45 s a step on the smoke's eight rows (44,500 tokens a step, about 1.0 ms a token) and 1.6 s a validation
  row.

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

## Estimates (mine, from the smokes' rates)

Prompts are 9,956 tokens with their tools at the median (the row gate's count on the smoke rows), about 145
million tokens over the 14,823 rows, so about 39,000 tokens a step. At the smoke's rate on an H200 (1.0 ms a
token) that is 40 s a step and 41 hours for the 3,706 steps, plus 28 validations of 7 minutes: about 44 GPU-hours.
At the rate the complete-evidence clone's smoke measured on an RTX PRO 6000 (1.3 ms a token) it is about 57. The
A100 and H100 are not measured. `submit_cell.sh` counts 60 s a step and submits 7 links of 12 hours, which covers
both; a pass that ran short would resume with one more link.

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
  **Shown 2026-11-16 on the second run** (the first ran out of memory before its first step).
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

Before the full run every stage had run with the model on the smoke's sizes only: eight training rows, 32
validation rows, eight evaluation cases. The row gate has since passed on all 15,079 rows, and the pass has been
preempted, requeued and resumed from its newest checkpoint 24 times (above), on A100 cards among them, and one
link has handed over to the next at the 12-hour limit, and the pass has ended with its adapter and receipt. Not
exercised yet: the queued links leaving at once after it, the evaluation on all 300 cases, and training on an
H100.
