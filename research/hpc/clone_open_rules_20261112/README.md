# Open-rules clone (2026-11-12)

A supervised pass of the student on the open-rules selection, and the baselines that read it: S0 and the new
adapter, greedy, on the RL series' 300 development cases with the controller's five rules lifted (`open_rules`).
Work directory on torch: `/scratch/yx3882/clone_open_rules_20261112` (`$WORK`). **Prepared only: nothing is
committed, deployed or submitted, and each step needs the user's word.**

## Why

Built from the known error configuration, corrective sequences solve cases the rule expert cannot, but only with
controller rules lifted that the student never saw lifted. S0 and every RL arm so far were trained and measured with
the rules in place. Before an RL arm in the open environment, two things are measured:

1. **S0 in the open environment.** Its prompt now names the lifted rules and its commit tool may keep a rejected
   candidate; it was never trained on either. This is the baseline of any RL arm in that environment, read against arm
   A's baseline of S0 with the rules in place.
2. **A start adapter cloned in the open environment.** S0's own cases, rendered again with the rules lifted, plus
   truth-built sequences on the cases the demonstration fails.

## The runs

All runs use the RL driver's baseline phase with greedy decoding, bf16, 40 actions, bounded history, a prompt limit of
49,152 tokens and seed 20261107, on the 300 cases of `ieee14_dev300.json` (sha256 `04e2ad09...`).

| Run | Adapter | Rules | Config | Differs from |
| --- | --- | --- | --- | --- |
| S0, rules in place | S0 | in place | arm A of the RL series (measured) | |
| S0, rules lifted | S0 | five lifted | `configs/baseline_s0_open.json` | arm A in `environment_options` only |
| Clone, rules lifted | this pass's | five lifted | `configs/baseline_clone_open.json` | `baseline_s0_open` in `initial_adapter` only |
| Smoke | S0 | five lifted | `configs/baseline_smoke_open.json` | `baseline_s0_open` in `development_suite` only (8 cases) |

`baseline_config.py` writes the configs from arm A's (`make_configs.arm_config`); `test_cell.py` checks each
difference. `environment_options.open_rules` names every rule (not `1`), so a rule added to the option later is not
lifted here without a word.

**What the open baselines are read against**, on the same 300 cases:

| | Solved of 300 | Note |
| --- | --- | --- |
| S0, rules in place (arm A's baseline) | 182 | 168 clean, 7 declines in 563 decisions, 46 wrong targets committed, 12.85 mean actions |
| Rule expert (guarded, the student's environment) | 184 | local; its episodes are the same with the rules lifted (all 300) |
| Rule expert, route variant (the series' reference) | 188 | `teacher_map.json` |
| Truth-built sequence, rules lifted | 247 | reads the truth: a labelled upper bound. 184 of the 184 the rule expert solves, 63 of the 116 it fails; 11.12 actions against the rule expert's 12.97 |

The S0 row is arm A's baseline as the complete-history clone's README records it. The truth-built reference is in
`output/compound_rl_20261106/truth_reference_open_rules_dev300` of the main checkout (`truth_sequences --no-rows`).

The driver's evaluation records now carry, where rules are lifted, an `open_rules` block per episode: how often each
lifted rule was needed, and every rejected or inconclusive candidate kept or rolled back with the audit's label
(`psse_env.rl.lifecycle.rejected_candidate_decisions`). `evaluation.json` sums them (`open_rules.rejected_candidates`:
true and wrong targets kept and rolled back). A rejected candidate kept on a healthy component counts in the reward's
audit of kept wrong targets, as a wrong accepted one does. With no rule lifted the records are as before.

## The pass

| | S0's pass (2026-11-06) | This pass |
| --- | --- | --- |
| Start adapter | R1 of the conventional cell | the same copy (`compound_cloning_20261106/data/start_adapter`) |
| Training rows | 8,866 of 713 cases | 10,966 of 972 cases |
| Validation view | 192 rows of 44 cases | 192 rows of 56 cases (746 held-out rows) |
| Rules in the rows' environment | in place | five lifted |
| Optimizer | lr 3e-5, micro-batch 1 x 4 accumulation, one pass, LoRA rank 16, 4-bit base, seed 20261107 | the same, option for option (`test_cell.py`) |
| Tool list in the prompt | the run-time list of each row | the same |
| Row limit | 32,768 tokens | the same (longest row about 37,100 characters, about 13,300 tokens) |
| Optimizer steps | 2,217 | 2,742 |

The selection (`research/compound_errors/open_rules_selection.py`, settings in `selection_report.json`):

| Part | Cases, train + held out | Training rows |
| --- | --- | --- |
| Demonstration, whole cases: S0's own cases, rendered again with the rules lifted | 560 + 37 | 6,668 |
| Truth-built sequences: the demonstration fails the case, the sequence passes | 412 + 25 | 3,916 |
| Demonstration roll-backs of wrong candidates kept in 153 truth-built cases S0 had | | 382 |

- **Same rows as S0:** 544 of the 560 demonstration cases carry exactly the rows S0 was trained on. 19 of S0's rows
  are left out, roll-backs of a rejected true target, because keeping such a candidate is the right step once the
  commit rule is lifted.
- **Keep-or-roll-back rows in training:** 451 accepted wrong candidates rolled back (declines), 443 rejected wrong
  candidates rolled back, 307 rejected true candidates kept, 268 corrections that needed a lifted rule.
- **Left out of the truth-built part:** corrections whose target the row's prompt does not name, and the 5 roots
  reserved by the RL series (`reserved_roots.json`, the fork probe's) that S0's pass never had. The 17 reserved roots
  S0's pass had stay.
- **Prompts:** every row's system prompt, user message and tool list is what the run-time policy renders for the
  row's own state (`_CanonicalResearchNativePolicy.render_messages`), checked on all 11,158 rows
  (`checks/prompt_parity.json`).

Rebuild, from the worktree, with the main checkout's `output/`:

```bash
O=/c/Users/Holiday/Documents/ChatGPT/PSSE_Agent/output/compound_rl_20261106
python -m research.compound_errors.truth_sequences --suite $O/train_ieee14/suite.json \
    --output-dir $O/truth_rows_open_rules_ieee14 --reference $O/truth_probe_train_ieee14/results.jsonl --workers 22
python -m research.compound_errors.commit_reader_loop run --readers <readers.pkl> --level full --budget 0.01 --decider audit \
    --suite $O/train_ieee14/suite.json --output-dir $O/cloning_open_rules_ieee14 --export-rows --workers 22 \
    --env-option detailed_evidence=1 --env-option pair_refit_adjacent=1 --env-option route_comparison=1 --env-option open_rules=1 \
    --expert-option abstain_isolated_lines=1 --expert-option state_ambiguity=1
python -m research.compound_errors.open_rules_selection --truth-rows $O/truth_rows_open_rules_ieee14/rows.jsonl \
    --truth-episodes $O/truth_rows_open_rules_ieee14/episodes.jsonl --demonstration-rows $O/cloning_open_rules_ieee14/rows.jsonl \
    --demonstration-episodes $O/cloning_open_rules_ieee14/episodes.jsonl \
    --demonstration-cases $O/open_rules_selection_ieee14/cloning_pass_cases.json \
    --reserved-roots research/hpc/rl_improvement_20261110/reserved_roots.json --thin run_wls=0.25 \
    --output-dir $O/open_rules_selection_ieee14
```

The demonstration run uses the options of the stored `cloning_ieee14_v2` run, which made S0's rows; all 2,550 of its
episodes are identical to that run's. `checks/` beside the views holds the local checks: `row_checks.json` (the row
gate's message checks), `prompt_parity.json`, and the scripts that wrote them.

## Files

| File | What it does |
| --- | --- |
| `cell.env` | Paths, settings, the job environment; sourced by the stages |
| `baseline_config.py`, `configs/*.json` | The three baseline configs, from arm A's (`--check` compares the files on disk) |
| `rows_manifest.py` | The rows manifest; refuses views that are not the ones the selection report and the row checks describe |
| `stage_row_gate.sbatch` | The tokenizer row gate over the full views on a CPU node; its result is cached for the pass's restarts |
| `stage_clone_train.sbatch` | The pass, or with `CLONE_MODE=smoke` about 200 rows for 2 steps, the longest row first |
| `stage_baseline.sbatch` | The driver's baseline phase from `configs/baseline_$BASELINE_NAME.json` |
| `submit_cell.sh`, `status_cell.sh` | Submit chains; show queue, receipts, losses and the baselines beside arm A's. Through ssh stdin |
| `deploy_from_windows.sh` | Source (its own clone of the RL series' source), rows with a manifest, configs. Submits nothing |
| `test_cell.py` | `python -m pytest research/hpc/clone_open_rules_20261112/test_cell.py`, on its own: it shares its file name with the other cells' tests |

The trainer entry (`train_runtime_tools.py`) and the smoke inputs (`smoke_inputs.py`) are the bounded cell's, used as
they are from the same source.

## Deploy, smoke, submit

From the worktree in Git Bash, SSH master up (`scripts/start_torch_ssh_master.ps1`), sources committed on the branch.

```bash
bash research/hpc/clone_open_rules_20261112/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
C=research/hpc/clone_open_rules_20261112
export MSYS_NO_PATHCONV=1
wsl -- ssh torch bash -s -- smoke < $C/submit_cell.sh   # 2 training steps on the longest rows; S0 on 8 cases with the rules lifted
wsl -- ssh torch bash -s < $C/status_cell.sh
wsl -- ssh torch bash -s -- s0 < $C/submit_cell.sh      # S0's baseline with the rules lifted (needs no pass)
wsl -- ssh torch bash -s -- pass < $C/submit_cell.sh    # the row gate, then the pass (4 links)
wsl -- ssh torch bash -s -- clone < $C/submit_cell.sh   # the clone's baseline, after the pass's last link
```

`all` submits S0's baseline and the row gate, pass and clone's baseline at once. Read after the smoke:
`out/smoke/gate_timing.json` (row lengths by the processor), the log's seconds per step, `gpu-util-*` (peak memory),
and `out/baseline_smoke_open/baseline/evaluation.json` (the `open_rules` block, prompt refusals).

## Duration and GPU-hours

Estimates; nothing here has run on a GPU.

- **The pass: about 16 to 27 GPU-hours.** S0's pass took 12.5 GPU-hours on H200s for 2,217 steps, about 20 s per
  step, and 34 s per step was measured on an A100. These rows are about as long, and there are 2,742 steps.
- **Each baseline: about 6 GPU-hours** (arm B's took 6.1 hours of running), so about 12 for the two.
- **Row gate:** a few CPU minutes (239 s for S0's views). **Smoke:** under 1 GPU-hour.

## Not verified without the cluster

- The tokenizer row gate on these rows. The local checks are its message checks only.
- How S0 behaves when its prompt names rules it never trained with lifted; the smoke's 8 cases are the first look.
- The driver's new `open_rules` record on a real episode (tested on stand-ins, `psse_env/rl/test_train.py`).
- Memory and seconds per step.

## Not built

- The RL update's lifecycle term and decision credit label decisions on accepted candidates only, and forks branch
  commit against decline only. Keeping or rolling back a rejected candidate is measured in the evaluation, not
  trained. That matters for an RL arm in the open environment, not for these baselines.
- No run in the closed environment from the clone, and no conventional-suite check of it.
- The complete-history clone (`research/hpc/clone_complete_20261112`, another worktree, uncommitted) also edits
  `research/compound_errors/cloning_rows.py`; merging the two branches needs a look there.
