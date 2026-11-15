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
| `stage_arm.sbatch` | The RL arm: the driver's iterations from `configs/baseline_clone_open.json`, in the run its baseline began |
| `submit_cell.sh`, `status_cell.sh` | Submit chains; show queue, receipts, losses, the baselines beside arm A's and the arm's phases. Through ssh stdin |
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
wsl -- ssh torch bash -s -- smoke_baseline < $C/submit_cell.sh   # the baseline smoke alone, again; an earlier one is moved aside
wsl -- ssh torch bash -s < $C/status_cell.sh
wsl -- ssh torch bash -s -- s0 < $C/submit_cell.sh      # S0's baseline with the rules lifted (needs no pass)
wsl -- ssh torch bash -s -- pass < $C/submit_cell.sh    # the row gate, then the pass (4 links)
wsl -- ssh torch bash -s -- clone < $C/submit_cell.sh   # the clone's baseline, after the pass's last link
wsl -- ssh torch bash -s -- arm < $C/submit_cell.sh     # the RL arm from the clone (6 links, target 2 iterations); see "The RL arm"
```

`all` submits S0's baseline and the row gate, pass and clone's baseline at once. Read after the smoke:
`out/smoke/gate_timing.json` (row lengths by the processor), the log's seconds per step, `gpu-util-*` (peak memory),
and `out/baseline_smoke_open/baseline/evaluation.json` (the `open_rules` block, prompt refusals).

## Duration and GPU-hours

Measured in the smokes of 2026-11-13 on an RTX PRO 6000 (97 GB); the rest estimated.

- **The pass: about 21 GPU-hours on that card.** 25 to 27 s per optimizer step, so about 20 hours for 2,742 steps,
  plus about 1.5 hours for the 21 validation passes (42 s per 32 rows). By card, about 16 to 27 GPU-hours: S0's pass
  took 12.5 GPU-hours on H200s, and 34 s per step was measured on an A100. Peak memory 53 GB.
- **Each baseline: about 6 GPU-hours** (arm B's took 6.1 hours of running), so about 12 for the two. The smoke's 8
  cases took 6.5 minutes on two shards, peak 58.5 GB.
- **Row gate: 4 minutes** (job 22955974: 11,158 rows in 236 s, 4 min 17 s in all), after about 8 hours in the CPU
  queue. The smoke's 232 rows took 68 s, about 41 s of it start-up; the stage allows 3 hours. **Smokes:** about
  6 minutes each.

## Smokes (2026-11-13)

Both on source 68b1950, one RTX PRO 6000 each.

- **Training smoke** (job 22943411): 200 rows, the longest first, 2 optimizer steps. Row gate 232 rows in 68 s,
  longest row 12,939 tokens, nothing cut; validation loss 0.218 after the 2 steps; the reload canary passed, and a
  resume would start from checkpoint-2.
- **Baseline smoke** (job 22943412): S0 with the five rules lifted solves 7 of the 8 cases, the same 7 as arm A's S0
  with the rules in place, with 3 clean solves against 6. In r0_007c37519b53, r0_018a7be478a1 and r0_0681aebdc72f
  the last hand-off was refused, sent again and cut as a loop. Replaying arm A's actions without the model shows that
  the open environment accepts the usual hand-off in all three, and that a request text the audit does not support
  gives exactly the run's record (`operator_escalation_request_unsupported`, then a repeated-failure stop). The
  record kept tool names only, so S0's text is not known; two open-rules sentences said "request operator review".
  They now name the tool and the request (`dataset_builder.OPEN_RULES_HANDOFF`), the views were re-rendered with them
  (`selection_report.json`, "rerendered"; the old views in `superseded_20261113_handoff_wording/`), and the driver's
  records keep each refused step's error and every hand-off's request. One case used a lifted rule:
  r0_18ba2a24188a kept a rejected true line-4 candidate and a rejected wrong meter-48 candidate, and failed, as it
  does with the rules in place.
- **Baseline smoke again** (job 22951318, source 851a38f, reworded sentences; the first one's output is
  `out/baseline_smoke_open.before_20261113T051846Z`): 7 of 8 solved and 6 clean, as arm A on the same cases; every
  hand-off sends `operator_escalation:recovery_options_exhausted`; 1 refused step (a parameter comparison in
  r0_243985153e00, refused in arm A too); no loop; 12.4 actions on average against arm A's 13.0. The three cases of
  the refused hand-offs now take arm A's actions one for one. r0_18ba2a24188a is as before, so 4 cases count a
  corrupted healthy component against arm A's 3 (the other three are breaker cases flagged in both).

## S0's baseline with the rules lifted (job 22955971, 2026-11-13)

Source 851a38f, 300 cases, one preemption, done 2 h 33 min after its last start. Read case by case against arm A's
baseline (S0 with the rules in place) and split by the rule expert's outcome, from the truth reference's labels:

| Cases | Arm A solved | Lifted solved | Gained | Lost |
| --- | --- | --- | --- | --- |
| Rule expert solves (184) | 180 | 180 | 3 | 3 |
| Rule expert fails, truth-built passes (63) | 2 | 6 | 4 | 0 |
| Rule expert fails, truth-built fails (53) | 0 | 0 | 0 | 0 |
| All (300) | 182 | 186 | 7 | 3 |

| | Arm A | Lifted |
| --- | --- | --- |
| Clean solves | 168 | 178, or 176 when a kept rejected candidate does not count as clean |
| Wrong corrections kept | 46 | 62: 50 accepted ones and 12 rejected ones |
| Episodes with a corrupted healthy component | 47 | 52 |
| Refused actions | 121 | 84 |
| Loop episodes | 23 | 14 |
| Mean actions | 12.85 | 12.78 |

- 224 of the 300 tool sequences are identical. A lifted rule is used in 19 episodes.
- The 3 losses are all cases the rule expert solves, and all come from `correction_after_estimator_passes`: once
  the WLS passed, S0 corrected a healthy meter and kept it.
- The gains: 3 of the 63 by `correction_outside_context` (a meter correction after the line correction, on a meter
  the context did not list); 2 by keeping a rejected candidate on the true line (one in each split); 2 by different
  choices without a lifted rule.
- Rejected candidates: 93 decisions; kept 6 of 17 on a true target and 12 of 76 on a wrong one. The 12 wrong ones
  kept all sit in unsolved episodes.
- Hand-offs: all 310 send `operator_escalation:recovery_options_exhausted`. 12 were refused
  (`run_wls_successful_evidence_missing`), each sent again and stopped as a loop; 4 of the 12 are still solved. In
  all 12 the last check before the hand-off was `verify_candidate`, and the hand-off audit accepts only a
  successful `run_wls` on the active state (a rule not lifted). Arm A shows the same pattern in 12 episodes (2 of
  them the same cases), so it is S0's habit, not an effect of the lifted rules.

## Row gate on the full views (job 22955974, 2026-11-13)

All 11,158 rows pass the tokenizer gate with their run-time tool lists: none cut, none without a supervised token.
Prompt lengths in tokens, against the 32,768 limit:

| View | Rows | Median | p95 | p99 | Longest |
| --- | --- | --- | --- | --- | --- |
| Training | 10,966 | 5,787 | 9,492 | 11,236 | 12,992 |
| Validation | 192 | 6,220 | 10,356 | 11,978 | 12,269 |

Targets of the training view: commit 1,890, correct parameters 1,437, measurement context 1,339, parameter context
1,191, hand-off 972, WLS 942, rollback 894, topology context 772, correct topology 639, correct measurements 545,
parameter comparison 345. None targets `get_verification_snapshot`, the canonical name of `verify_candidate`, so
nothing here teaches the habit behind S0's 12 refused hand-offs; S0's start adapter, R1, is also this pass's.

The pass's first link (job 22955975) started at 14:13:47Z on an A100 80 GB (node ga029): 2,742 optimizer steps,
saving every 16 and validating every 128.

## The pass and the clone's baseline (jobs 22955975 and 22955981, 2026-11-14)

The pass ran from 2026-11-13 14:13:47Z to 2026-11-14 12:23:24Z under one job id, with 17 preemptions: 2,742 of 2,742
steps. Training loss over the last 64 steps 0.0163. Validation loss 0.0537 at step 128, 0.0253 at step 1,152, 0.0191
at the end. The clone's baseline ran 12:30:37Z to 15:36:56Z on one RTX PRO 6000 with no preemption.

The clone on the 300 development cases with the rules lifted, beside S0 with the rules lifted and S0 with the rules
in place (arm A's baseline):

| | S0, rules in place | S0, rules lifted | Clone, rules lifted |
| --- | --- | --- | --- |
| Solved | 182 | 186 | 163 |
| Clean solves | 168 | 178 | 155 |
| Clean, no rejected candidate kept | 168 | 176 | 148 |
| Wrong corrections kept | 46 | 62 | 110 |
| Episodes with a wrong correction kept | 32 | 38 | 75 |
| Episodes with a corrupted healthy component | 47 | 52 | 90 |
| Declines of an accepted candidate | 7 | 3 | 37 |
| Loop episodes | 23 | 14 | 4 |
| Refused actions | 121 | 84 | 94 |
| Mean actions | 12.85 | 12.78 | 13.48 |
| Episodes using a lifted rule | 0 | 19 | 93 |

| Cases | S0, rules in place | S0, rules lifted | Clone, rules lifted |
| --- | --- | --- | --- |
| Rule expert solves (184) | 180 | 180 | 143 |
| Rule expert fails, truth-built passes (63) | 2 | 6 | 20 |
| Rule expert fails, truth-built fails (53) | 0 | 0 | 0 |

The clone does what the selection was built for on the 63 cases only truth-built sequences solve: 20 against 6. It
loses 37 of the cases the rule expert solves. Against S0 with the rules lifted it loses 40 cases and gains 17; 166 of
the 300 tool sequences are identical. Of its 37 declines, 29 are of wrong targets and 8 of true ones.

The clone's episodes by the lifted rule they used:

| Lifted rule | Episodes | Clone solves | S0 lifted solves | Rule expert solves | Clone corrupted |
| --- | --- | --- | --- | --- | --- |
| `correction_after_estimator_passes` | 55 | 2 | 31 | 34 | 52 |
| `correction_outside_context` | 26 | 8 | 9 | 7 | 13 |
| `commit_rejected_candidate` | 17 | 7 | 5 | 4 | 9 |
| `correction_on_closed_route` | 8 | 5 | 3 | 3 | 6 |

**One habit costs 31 of the 40 lost cases: a meter correction after the estimator already passes.** S0 ends such a
case `... commit, measurement context, hand-off`; the clone goes on `... commit, measurement context, correct
measurements, WLS, commit, measurement context, hand-off`. The clone made 58 corrections after a pass:

| Kind | Count | On a truly faulted component | Verdict | Kept | In solved episodes |
| --- | --- | --- | --- | --- | --- |
| Meter | 54 | 0 | 54 accepted | 54 | 0 |
| Line parameters | 4 | 3 | | | 2 |

No training row teaches the meter correction. The first reading, that the 61 truth-built rows using
`correction_after_estimator_passes` teach it, was wrong:

- Those 61 rows are 57 second-line corrections and 4 breaker corrections, none a meter. The parameter context names
  the target with a multiplier between 3.1 and 5.9, median 4.3: below the alarm, so the rule expert passes it by. After
  a commit with the estimator passing, the WLS output lists a multiplier of 3 or more in 63 of the 97 states where the
  truth-built sequence goes on, against 98 of the 880 states where it wraps up. They stay in the selection.
- All 880 rows at a measurement context with the estimator passing hand off, 222 of them with a meter correction on
  offer at a residual near 3.
- Every meter correction in the rows has a strong alarm: the largest normalized residual has a median of 17 to 20.

So the habit is the clone's own generalization, from lines and breakers to meters: after a pass, take a weak
correction the context offers. The verifier accepts each one, because the estimator still passes afterwards. That
reading fits the counts and is not tested directly. Scripts and outputs: the session's scratchpad
(`clone_open_read.py`, `postpass_table.py`, `postpass_sep.py`, `postpass_dev.sh`).

## The RL arm from the clone (prepared 2026-11-14, not submitted)

**What it is.** Arm A's update, rewards and root draws, in the open environment, from the cloned adapter. It is the
run the clone's baseline began: `configs/baseline_clone_open.json`, run folder `out/baseline_clone_open`, manifest at
the first rollout, baseline 163 of 300. So every evaluation pairs with that baseline and nothing is measured twice.
The config differs from arm A's in `environment_options` and `initial_adapter` only.

**Why it should have signal where arms A and C had little.** The meter habit is the policy's own error, frequent and
always penalized: 54 of 300 greedy episodes end with a healthy meter corrupted, in cases a hand-off solves. Three
terms reach it:

- the episode reward: of a root's four samples, one that hands off solves the case and one that corrects is charged
  0.5 and solves nothing;
- the supervised lifecycle term: each kept meter correction is an accepted candidate on a wrong target, so its
  decision gets a row whose completion is the rollback;
- the invalid-action and loop charges, as before.

No decision credit: it replaces the supervised row at the states it covers and gives a confident choice little
gradient, which in arm C held the push back. The config can take it later (`decision_credit`), at about 2 GPU-hours
per iteration.

**What changed in the RL code for it** (`psse_env/rl`, with tests):

- Where the commit rule is lifted, keeping or rolling back a rejected candidate is the policy's choice, so it is now
  a labelled decision: `lifecycle_decisions(..., rejected_choice=True)`, set by the driver from the config
  (`train.rejected_choice`). The label is the audit's, as for accepted candidates and as the truth-built rows have it:
  keep a candidate on a truly faulted component, roll back any other. The supervised and the sampled lifecycle terms
  train these decisions with the others. With the rule in place nothing changes.
- A sampled episode or a branch that keeps a rejected candidate on a healthy component is charged as one that keeps
  a wrong accepted one. The evaluation already did this.
- Reports read the two kinds apart: the counts under `lifecycle` stay those of accepted candidates;
  `rejected_candidates` appears in the rollout summary, the score summary and the update's decision diagnostics;
  the row report counts `supervised_rows_on_rejected_candidates`.
- Not changed: forks and the decision credit branch accepted candidates only. Whether to correct at all after a
  pass, or outside the context, is not a two-way choice and has no label; the episode reward carries it.

**Checked without a GPU.**

- The RL package's tests and this cell's tests pass.
- Both choices can be rendered at every candidate of the selection: 2,866 rows whose target commits or rolls back,
  330 of them keeping a rejected candidate. Each tool is visible on the observation and its call passes the policy's
  parser and schema check. A labelled choice that cannot be rendered stops an update, so this is the check that
  matters before a rollout is spent. The chat-template half of that path is the one arms A and C ran with none
  refused; this machine has no tokenizer for it.
- Eight truth-built sequences that keep a rejected candidate, replayed in the real environment with the rules lifted:
  each kept candidate is among the episode's decisions, in order, labelled as agreeing, and the accepted decisions
  are the ones the closed reading gives.

**Not verified.** No update has run in the open environment. The first update's report is the test:
`integrity.ok`, `rows.target_renderer.refused` 0, and `rows.supervised_rows_on_rejected_candidates` above 0.

**Deploy and submit**, each on the user's word, sources committed, SSH master up:

```bash
bash research/hpc/clone_open_rules_20261112/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch bash -s -- arm < research/hpc/clone_open_rules_20261112/submit_cell.sh
```

The deploy moves the cell's source to the new commit; the rows and configs are already there and are skipped by
checksum. `arm` submits six linked jobs named `orc-arm` with a target of two iterations, the series' screen. It
refuses a source without the labels above, a baseline that has not closed, and a queued `orc-arm`. A third iteration
is a new submission (`arm 4 3`), made to resolve a concrete uncertain signal.

**Cost, estimated from arm A's phases:** two iterations are about 27 GPU-hours of work, 30 to 35 with restarts: a
rollout about 3.3 hours, an update 6.5 to 7.2, an evaluation 3.1 to 3.9. The clone's episodes are 5% longer.

**What to read.**

| After | File | Read |
| --- | --- | --- |
| First rollout | `iteration_000/rollout_summary.json` | Success rate, groups with reward spread, wrong targets committed and declined, `rejected_candidates` |
| First update | `iteration_000/update_report.json` | The three fields above; the share of the gradient on supervised rows |
| Each evaluation | `iteration_00N/evaluation_episodes.jsonl` | Paired with the baseline: corrections after a pass by kind, wrong corrections kept, solved by the rule expert's split |

The arm has done its job when the 54 meter corrections after a pass fall and the 143 of the rule expert's 184 rise,
without the 20 of the 63 falling. A decline rate that rises on true and wrong targets alike is the failure arms A
and C showed.

## Not built

- Forks and the decision credit branch accepted candidates only. Keeping or rolling back a rejected candidate is
  labelled and trained since 2026-11-14 (see "The RL arm"), not branched.
- The choice to correct after the estimator passes, or outside the context, has no label of its own; only the
  episode reward reaches it.
- No run in the closed environment from the clone, and no conventional-suite check of it.
- The complete-history clone (`research/hpc/clone_complete_20261112`, another worktree, uncommitted) also edits
  `research/compound_errors/cloning_rows.py`; merging the two branches needs a look there.
