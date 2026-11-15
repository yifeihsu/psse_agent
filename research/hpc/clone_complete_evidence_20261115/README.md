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

One optimizer step takes four rows. The rows are about half as long again as the open-rules rows (see the numbers
in `render_manifest.json`; token counts there are a sibling tokenizer's estimate), so the pass is estimated at 22 to
38 GPU-hours against 16 to 27, and `submit_cell.sh` counts 60 s a step when it sizes the chain.

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
