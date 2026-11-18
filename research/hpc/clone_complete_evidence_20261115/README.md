# Complete-evidence clone (2026-11-15)

A cloning pass on rows in which the policy reads the estimator's output whole, and a baseline of the cloned adapter
on the 300 development cases. Prepared on the user's word of 2026-11-15. Nothing here is deployed or submitted
without the user's word for that step.

Work directory on torch: `/scratch/yx3882/clone_complete_evidence_20261115` (`$WORK`), with a source of its own.
The open-rules clone's cell (`/scratch/yx3882/clone_open_rules_20261112`) and the RL series are read in place and
never moved.

## Why

The clone of 2026-11-12 solves 163 of the 300 development cases. The rule expert solves 184 to 188 and the
truth-built sequences 247. The user's direction of 2026-11-15: the evidence that clone reads is what the rule
expert's screens kept of each solve (the five largest entries above a threshold, the lines a dominance test admits,
the breakers whose flip clears a bar, a list of supported corrections), and a policy held to that digest cannot pass
the rule expert. The policy should read the estimator's output and decide.

## What the policy reads here

Environment option `complete_evidence` (`psse_env/providers/complete_evidence.py`, off by default):

- every bus and line of a solve, with the estimated voltage and angle, signed normalized residuals and both
  multipliers, kept in view while the case is active;
- every measurement with its index, value and residual; every line with its R, X and multipliers; every breaker
  with its status, multipliers, what a flip would do and, for the eight the estimator re-estimates, the chi-square
  after the flip;
- none of the screens' conclusions (`dataset_builder.RULE_SCREEN_KEYS`): supported corrections, route status,
  top-five lists, dominance and ambiguity verdicts, confidences, the open-terminal cue, the trial screening of
  proposed flips, the signature tags, and the line comparison's own ranking.

The controller's five rules are lifted as in the open-rules cell. The line comparison
(`compare_parameter_candidates`) takes any two or three lines the policy names: the tool came from the rule expert
and held its lines to the context's supported list (the user's word of 2026-11-15; commit ce7d3fb).

Left in the view: the estimator's alarm, the verifier's verdict on a candidate beside its numbers, what a correction
wrote, the bounded history (8 events, 4,096 characters).

## The rows

`output/compound_rl_20261106/complete_evidence_selection_ieee14` in the main checkout (not in git):
`selection.train_view.jsonl`, `selection.validation.jsonl`, `selection_report.json`, `render_manifest.json`,
`checks/`. The deploy writes `manifest.json` there.

They are the open-rules selection's cases and rows, rendered again: both row pipelines run with
`--env-option complete_evidence=1` and otherwise the stored options, then the same selection command. Every episode
and every target is the same as in the open-rules rows. The train view holds every row of the open-rules selection
plus 95 truth-built corrections whose component the old prompt did not name (the selection dropped those as
unnamed; here every component is in a table).

Rebuild, from the worktree, with the main checkout's `output/`:

```bash
O=/c/Users/Holiday/Documents/ChatGPT/PSSE_Agent/output/compound_rl_20261106
python -m research.compound_errors.truth_sequences --suite $O/train_ieee14/suite.json \
    --output-dir $O/truth_rows_complete_evidence_ieee14 --reference $O/truth_probe_train_ieee14/results.jsonl \
    --workers 26 --env-option complete_evidence=1
python -m research.compound_errors.commit_reader_loop run --readers <readers.pkl> --level full --budget 0.01 --decider audit \
    --suite $O/train_ieee14/suite.json --output-dir $O/cloning_complete_evidence_ieee14 --export-rows --workers 26 \
    --env-option detailed_evidence=1 --env-option pair_refit_adjacent=1 --env-option route_comparison=1 \
    --env-option open_rules=1 --env-option complete_evidence=1 \
    --expert-option abstain_isolated_lines=1 --expert-option state_ambiguity=1
python -m research.compound_errors.open_rules_selection --truth-rows $O/truth_rows_complete_evidence_ieee14/rows.jsonl \
    --truth-episodes $O/truth_rows_complete_evidence_ieee14/episodes.jsonl \
    --demonstration-rows $O/cloning_complete_evidence_ieee14/rows.jsonl \
    --demonstration-episodes $O/cloning_complete_evidence_ieee14/episodes.jsonl \
    --demonstration-cases $O/open_rules_selection_ieee14/cloning_pass_cases.json \
    --reserved-roots research/hpc/rl_improvement_20261110/reserved_roots.json --thin run_wls=0.25 \
    --output-dir $O/complete_evidence_selection_ieee14
```

`checks/` beside the views holds the local checks and the scripts that wrote them: `row_checks.json` (the row
gate's message checks, and for every row the contract, the paragraph, no screen's conclusion, the two schemas),
`prompt_parity.json` (the run-time policy renders every row's prompt character for character), and the comparisons
with the open-rules rows. The tokenizer part of the row gate runs on the cluster.

## The pass

S0's pass (`research/hpc/compound_cloning_20261106`) and the open-rules clone's, on these rows: the same start
adapter (that cell's copy of R1), the same trainer options one for one (lr 3e-5, 1 x 4, LoRA 16, 4-bit, seed
20261107, run-time tool lists, limit 32,768 tokens). No adapter trained since can read these rows, so the pass
starts where those two did.

One trainer option is added, `--supervised-logits-only` (`psse_env/sft/supervised_logits.py`): the loss is computed
from the logits of the target's span alone. It is the same loss. Without it the model projects every prompt position
onto the vocabulary, about 1 MB a position, and the first training smoke (job 23274632, 2026-11-15) ran out of a
95 GB card on the longest row, 20,390 tokens. The earlier passes ran without the option and are not changed.

With the option the training smoke passed (job 23285899, 2026-11-16, an RTX PRO 6000): all 41 forwards with labels
kept the span, at most 60 logits of 20,390 positions; the card's sampled peak was 28.6 GB; two steps, two
checkpoints, a resume would start from the second. The first steps look like the earlier passes' (gradient norms 33
and 90 against 19 to 140 there, token accuracy 0.94 and 0.96). The validation loss after two steps is 0.34 on 32
rows, against 0.21 and 0.22 in the two earlier smokes.

One optimizer step takes four rows. The rows are about half as long again as the open-rules rows (see the numbers
in `render_manifest.json`; token counts there are a sibling tokenizer's estimate), so the pass was estimated at 22
to 38 GPU-hours against 16 to 27, and `submit_cell.sh` counts 60 s a step when it sizes the chain. The smoke
measured 54 s a step on its eight rows (the longest first, 41,400 tokens a step) and 2.2 s a validation row. At that
rate the view's 34,800 tokens a step give about 35 GPU-hours for the 2,766 steps, and each of the 21 validations
takes 7 minutes: 37 GPU-hours in all on that card.

## The baselines

`baseline_config.py` writes two configs, each the open-rules clone's baseline config with `complete_evidence`:

| Config | Adapter | Cases | Read for |
| --- | --- | --- | --- |
| `baseline_clone_complete.json` | this cell's clone | the 300 development cases | the result |
| `baseline_smoke_complete.json` | S0 | eight of them | prompt length, memory, refusals |

S0 was trained on another prompt, so what the smoke solves says nothing.

The clone's baseline is read against, on the same 300 cases:

| Reference | Solved |
| --- | --- |
| S0, rules in place (arm A of the RL series) | 182 |
| S0, rules lifted | 186 |
| The open-rules clone, rules lifted | 163 |
| Rule expert | 184 to 188 |
| Truth-built sequences, rules lifted | 247 |

Read it by the split of the open-rules clone's result: the cases the rule expert solves (184) and the cases only the
truth-built sequences solve (63), with wrong corrections kept.

## Result (chain of 2026-11-16, read 2026-11-18)

The clone solves 119 of the 300 development cases, below every reference.

**The pass.** 2,766 of 2,766 steps, all in the first link (job 23289367, 30 preemptions), ended 2026-11-17 14:40Z;
the five later links exited at once. Training loss 0.023 over the last 64 steps. Validation loss on the 192 rows:
0.079 at step 128, 0.038 at 512, 0.031 at 2,048, 0.026 at 2,304, 0.029 at 2,766, with token accuracy 0.984 to 0.989
between steps 512 and 2,048. The open-rules clone ended at 0.019 on its rows (training 0.016). The reload canary was parseable
but not the row's action: the row compares lines 5 and 2, the adapter corrects line 5.

**The baseline** (job 23289373, two preemptions, 6.8 GPU-hours on RTX PRO 6000 cards, two workers, no prompt refused
for length, 51.3 million input tokens):

| On the 300 development cases | S0, rules in place | S0, rules lifted | Open-rules clone | This clone |
| --- | --- | --- | --- | --- |
| Solved | 182 | 186 | 163 | 119 |
| Of the 184 the rule expert solves | 180 | 180 | 143 | 101 |
| Of the 63 only the truth-built sequences solve | 2 | 6 | 20 | 18 |
| Of the 153 in classes with a breaker error | 107 | 109 | 92 | 50 |
| Of the 147 in classes without one | 75 | 77 | 71 | 69 |
| Solved with no refused action, loop or false commit | 168 | 178 | 155 | 54 |
| Wrong corrections kept | 46 | 62 | 110 | 453 |
| Episodes with a healthy component corrupted | 47 | 52 | 90 | 145 |
| Refused actions | 121 | 84 | 94 | 843 |
| Episodes cut as a loop | 23 | 14 | 4 | 83 |
| Episodes at the cap of 40 actions | | 2 | 7 | 70 |
| Mean actions | 12.8 | 12.8 | 13.5 | 21.1 |

Against the open-rules clone 72 cases are lost and 28 gained. The rule expert solves 106 of the 153 breaker cases
and 78 of the 147 others.

What the episodes show, this clone against the open-rules clone:

- Refused actions, 843 in 189 episodes: a parameter correction that fails 294; `finalize_diagnosis` 234, every call
  refused, a tool no row has as its target; a topology correction 165 (130 on a breaker the evidence does not
  support, 34 that change nothing); a hand-off 118; output that is no tool call 30.
- Candidates the verifier rejects: 448, kept 279. Kept on a true target 50 of 72, on a wrong one 229 of 376. The
  open-rules clone kept 20 of 74 (8 of 19 true, 12 of 55 wrong).
- Candidates the verifier accepts: kept 403 of 403 true and 224 of 245 wrong (519 of 527 and 98 of 127).
- First correction of an episode: a line 189, a breaker 85, a meter 25 (140, 129, 31).
- Line comparison: 11 calls (109). Corrections: line 901 (405), breaker 380 (178), meter 298 (220).
- A correction after the estimator passes: 14 actions in 11 episodes (58 in 55). The six gains looked at are all
  cases where the open-rules clone made that correction on a healthy meter.
- In the 132 episodes that use no lifted rule the clone solves 87; the open-rules clone solves 77 of the same cases.
  A correction outside the context's list is used in 137 episodes (23 solved), a kept rejected candidate in 126
  (17), a breaker on a closed route in 76 (2).

Not examined yet: the states at which the clone chose its components. That needs a local replay of the 300
episodes, as was done for the open-rules clone.

The 300 episode records, the read script and its output are in the worktree's `output/` (not in git):
`output/compound_rl_20261106/complete_evidence_clone_baseline_20261117/`.

## Cards

Both GPU stages name no partition, may be preempted and are requeued. The pass asks for
`a100|h100|h200|rtx6000|b200`, the baseline for `a100|h100|h200|rtx6000`.

The B200 is in the pass's list since 2026-11-17 (the user's word). Checked on torch that day: the scheduler takes the
list for this account (`sbatch --test-only`, which submits nothing), the jobs' torch build (2.10.0, CUDA 12.8) covers
the card, and the pass runs at 95% of a card with the model loaded and 27 GB, above the B200's low-utilization lines
(cancel below 70%, warning below 80%). The baseline is left off the B200: with two workers its smoke ran at 80% with
the model loaded. The clone's full baseline then ran at 95% on an RTX PRO 6000 (410 samples), so the smoke's figure
was the low one; the baseline's list was not changed.

The chain submitted on 2026-11-16 keeps the four-type list: a queued job's request cannot be changed on torch. The
wider list holds from the next deploy and submission. It adds cards to choose from. The seat is still a preemptible
one in the card type's base partition, so it does not lower the rate of preemptions.

## Files

| File | What it does |
| --- | --- |
| `cell.env` | Paths, settings, the job environment; sourced by the stages |
| `baseline_config.py`, `configs/*.json` | The two baseline configs (`--check` compares the files on disk) |
| `rows_manifest.py` | The rows manifest; refuses views that are not the ones the report and the row checks describe, and row checks made for another view |
| `stage_row_gate.sbatch` | The tokenizer row gate over the full views on a CPU node; its result is cached for the pass's restarts |
| `stage_clone_train.sbatch` | The pass, or with `CLONE_MODE=smoke` about 200 rows for 2 steps, the longest row first |
| `stage_baseline.sbatch` | The driver's baseline phase from `configs/baseline_$BASELINE_NAME.json` |
| `submit_cell.sh`, `status_cell.sh` | Submit chains; show queue, receipts, losses and the baselines beside the references. Through ssh stdin |
| `deploy_from_windows.sh` | Source (its own clone of the open-rules clone's source), rows with a manifest, configs. Submits nothing |
| `test_cell.py` | `python -m pytest research/hpc/clone_complete_evidence_20261115/test_cell.py`, on its own: it shares its file name with the other cells' tests |

The trainer entry (`train_runtime_tools.py`) and the smoke inputs (`smoke_inputs.py`) are the bounded cell's, used as
they are from the same source. There is no RL arm in this cell: an arm from this clone is a later submission.

## Order of work

Each step on the user's word.

```bash
bash research/hpc/clone_complete_evidence_20261115/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch bash -s -- smoke < research/hpc/clone_complete_evidence_20261115/submit_cell.sh
```

```bash
wsl -- ssh torch bash -s < research/hpc/clone_complete_evidence_20261115/status_cell.sh
```

```bash
wsl -- ssh torch bash -s -- all < research/hpc/clone_complete_evidence_20261115/submit_cell.sh
```

The smoke must show: every row through the gate with none cut, two optimizer steps and a resumable checkpoint, and
in the baseline smoke no prompt refused and the peak memory of the card. `all` submits the row gate, the pass and
the clone's baseline; `pass` and `clone` submit them one at a time. `smoke_train` and `smoke_baseline` run one half
of the smoke again.

## Not decided here

- An RL arm from this clone.
- The history: still 8 events in 4,096 characters, and older solves lose their tables.
- The flip re-estimation covers the eight breakers the estimator ranks first.
