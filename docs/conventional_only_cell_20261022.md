# Conventional-only confirmation cell (2026-10-22)

Cell `/scratch/yx3882/research_full_pipeline_20261022_conventional` on torch,
overrides `research/hpc/full_pipeline_20260907/overrides/conventional_only_20261022.env`,
source branch `claude/clever-roentgen-03b19c` (fast-forwarded from
`codex/classifier-triage-20261004` at 1193ff3). Research only.

## 1. Why

The PES GM 2027 manuscript (`docs/papers/pesgm2027`) reads its Table I from the
2026-10-07 classifier cell (`research_full_pipeline_20261007_classifier`,
`docs/classifier_triage_plan_20261004.md` section 11): on the 96 conventional
development roots the expert, BC0, R1 and R2 solve 95, 94, 95 and 95, with
invalid-action episodes 0/2/3/1, false-commit episodes 0/1/1/1, loop episodes
0/1/0/0 and 9.57/9.55/9.61/9.53 actions per episode. Two conditions of that
cell weaken the result, and Section VI of the manuscript says so: the policies
were trained on all 12 scenario families (with the high-impedance fault,
unbalance, harmonic and telemetry-only cohorts), and every WLS solve carried
the triage GNN's report, which could admit phasor and spectra requests. The
same 96 roots were also used throughout development, so they are not a sealed
test. Section VI specifies the confirmation: the conventional action set only,
auxiliary tools and the triage report removed for every policy, cloning and
DAgger on conventional families only, evaluation of every policy on a freshly
drawn, parent-disjoint test set with identical measurement realizations,
reported as success plus invalid-action, false-commit and loop episodes, mean
actions and solver calls. This cell is that protocol.

## 2. Design

| | 2026-10-07 classifier cell (Table I) | This cell |
| --- | --- | --- |
| Families in every plan | 12 | the 7 conventional: no_error, measurement, multi_measurement, parameter, topology, measurement+parameter, measurement+topology |
| Evidence profile | `classifier_gated_diagnostics`: triage report on every WLS solve; phasors and spectra after an admitted request or a fallback | `scada_only`: no auxiliary tool in the tool schemas or past the gate, no triage classifier, no triage report |
| Teacher | `baseline` | `baseline_compare`: the same baseline expert plus the bounded candidate comparison before an ambiguous parameter correction |
| D0 plan | 548 roots (370 conventional) | 370 roots: 30/60/50/60/60/60/50 |
| Rounds | 122 roots per round (74 conventional) | 74 roots per round: 6/12/10/12/12/12/10 |
| Development set | 160 roots, suite seed 20260911 (96 conventional: 8/16/12/16/16/16/12) | 192 roots, twice that composition (16/32/24/32/32/32/24), suite seed 20261022, fresh draw |
| Horizon, admission, meter floor | 40 actions, `recoverable`, 10 sigma | same |
| Parameter-ranking thresholds | training 1.2; development 1.0, true line within the top 2 | same |
| Seeds of the aggregate (also BC0's training seed) and the rounds | 20260910, 20260912 | same |
| Training recipe | Gemma 4 12B 4-bit, LoRA rank 16; BC0 one pass at 1e-4, best validation-loss checkpoint; rounds one pass at 3e-5 on the 1:1 mixture; beta 0.25 | same |
| Unverifiable expert label | abort the stage (none occurred) | end that episode and record it (`UNVERIFIED_EXPERT_LABEL_POLICY=stop`) |
| Round summaries | success, invalid actions, false commits, loops, steps | the same plus WLS calls and bounded comparisons per episode |

Family order in the counts above: no_error, measurement, multi_measurement,
parameter, topology, measurement+parameter, measurement+topology.

**The tool surface.** `psse_env/dagger/dataset_builder.py`
(`tool_schemas_for_observation`) gives a `scada_only` observation exactly the
schemas of `SCADA_ALLOWED_TOOLS` in `psse_env/evidence_profile.py`: WLS,
verification, the measurement, parameter and topology contexts,
`compare_parameter_candidates`, the three corrections, commit, rollback,
finalize and the operator request. The three-phase and harmonic contexts, the
three-phase NLM, HSE, both HIF estimators and `run_alternative_test` are
absent from the schemas and refused by the gate and the providers before any
provider call (`psse_env/test_scada_evidence_boundary.py`). The provider builds
no triage classifier under `scada_only` (`MatpowerDeploymentProviders` raises
when one is named), so no WLS solve carries a triage report, and the
observation sanitizer strips any `triage` or `gnn_screen` record as well.

**The teacher.** `ledger_ranked_compare`, the teacher of the 2026-10-21
comparison cell, follows the balanced hypothesis screen's ledger, and
`scada_only` runs no screen. `baseline_compare`
(`psse_env/oracle/expert_variants.py`) is the 2026-10-07 cell's baseline
expert with `candidate_comparison=True` and nothing else: when the fresh
parameter ranking does not dominate by 1.5 it compares the top two or three
lines on isolated branches before correcting one and follows the verdicts
(`docs/rl_experiments_20261014.md` section 12). Every other decision is the
baseline's, so the teacher differs from Table I's only on ambiguous parameter
rankings.

**Design decision: the node/breaker substation telemetry stays under
`scada_only` (2026-10-22).** Before this cell, `scada_only` stripped the
topology route's node/breaker channel along with the auxiliary streams: the
reported breaker map, the substation telemetry, the rendered operator layout
and meter nodes, the model identity, and the record a breaker correction leaves
for its own verification. Without them `get_topology_context` falls back to
bus-branch line-status flips. On the six topology and meter+topology roots that
come first in the 2026-10-07 development suite, the expert then solved five
with line-level corrections and failed the bus split
(r0_5691f69dd8f3, 24 actions, two wrong line flips and three rejected meter
edits). The 2026-09-24 check had found the same, 2 of 4 topology roots, both
failures bus splits. That would make the bus-split roots unsolvable for every
policy and the breaker-status cohorts incomparable with Table I, whose roots
were solved on the node/breaker route the manuscript describes (Sections II-C
and IV-A). The substation telemetry is SCADA-class data (node voltages, unit
and load meters, terminal and breaker flows at SCADA accuracies), not phasors
or spectra, so the profile now keeps exactly those seven fields
(`SCADA_NODE_BREAKER_FIELDS`); three-phase, harmonic, HIF, learned-screen and
triage data stay stripped. With them the expert solves all six roots with
breaker-level corrections (`cb_name`), the bus split in 8 actions. Dropping
only the correction record (`last_topology_correction`) leaves the node/breaker
context with no admissible breaker at all (six handoffs after 5 actions), so it
is part of the channel. Decided with the paper session on 2026-10-22 (option A
of two; option B, the unchanged profile, was rejected). The boundary tests
still prove that perturbing every auxiliary, simulator-truth, label,
learned-screen and triage field leaves state hashes, WLS results and the
expert's actions unchanged.

## 3. Preflight (local, CPU)

Everything below ran in this worktree on 2026-10-22 under the override's
settings (`PSSE_EXPERT_VARIANT=baseline_compare`, `scada_only`, local paths);
outputs under `output/conventional_only_20261022/` of the worktree
`.claude/worktrees/blissful-carson-36f26a` (gitignored).

**Topology probe (the design decision above).** The baseline and
`baseline_compare` experts on the first three topology and three
meter+topology roots of the 2026-10-07 development suite
(`output/classifier_triage_20261004/suites/development.json` of the main
checkout), re-executed under `scada_only`; script and records in
`topology_probe/` (`topo_probe.py`, `scada_only_{original,node_breaker}_{baseline,baseline_compare}.jsonl`,
`without_last_topology_correction.jsonl`):

| Root | Family | Original `scada_only` | With the node/breaker channel |
| --- | --- | --- | --- |
| r0_4ce17e11d6b6 | meter + topology | solved, 12 actions, line 18 out | solved, 12 actions, CB_Y1014_10N1_10N2 |
| r0_5691f69dd8f3 | topology (bus split) | **failed**, 24 actions, lines 15 and 9 out, three rejected meter edits | solved, 8 actions, CB_4_N1_N2 |
| r0_89c474e2d455 | topology | solved, 8 actions, line 11 out | solved, 8 actions, CB_6_L611_B2 |
| r0_d8f232bb6705 | topology | solved, 8 actions, line 13 out | solved, 8 actions, CB_6_L613_B1 |
| r0_ead064219e4f | meter + topology | solved, 12 actions, line 17 out | solved, 12 actions, CB_9_B1_N1 |
| r0_f6ad5ed787a7 | meter + topology | solved, 12 actions, line 17 out | solved, 12 actions, CB_9_B1_N1 |

Both teachers act identically on these roots (no parameter ranking is
involved). Without `last_topology_correction` alone all six end in a handoff
after 5 actions with no correction offered.

**Expert aggregate.** Fourteen roots (two per family) with the argv of
`stage_d0.sbatch` (`preflight_A/d0`, 49 s): 134 rows, none
quarantined, no unverifiable expert label. Every exported training row
carries exactly the 13 conventional tool schemas (`wls_from_path`,
`get_verification_snapshot`, the three contexts, `compare_parameter_candidates`,
the three corrections, commit, rollback, finalize, `ask_for_more_evidence`),
the system prompt's `scada_only` paragraph, and no `triage` field; the
demonstrations include two comparisons and four breaker corrections by
`cb_name`.

**Suite draw.** `build_suite.py` with the cell's plans, seed 20261022 and the
local aggregate as the D0 exclusion (`preflight_A/suite`, 13 min): 192
development roots (16/32/24/32/32/32/24) and 74 + 74 round roots
(6/12/10/12/12/12/10 each), 340 roots in all. Development strata: parameter
21 dominant, 10 ambiguous, 1 misranked; meter+parameter 26 dominant, 6
ambiguous. Topology: 25 isolated line terminals and 7 bus splits;
meter+topology: 24 isolated terminals. Multi-meter roots carry 2 to 5 bad
meters (5, 3, 5 and 11 roots with 2, 3, 4 and 5). The 182 injected meter
errors span 10.0 to 29.8 noise sigmas (median 13.1); the R/X factors lie in
[0.10, 0.50] or [2.04, 4.91]. `development.json` sha256 begins `fedf867685f2`.
The cluster's draw (section 4) is the test set; it differs from this local
one in part.

**Disjointness.** `check_suite_exclusion.py` over the local draw
(`preflight_A/disjointness_local.json`): no physical root shared between the
development set and either round, between the rounds, or with the D0
aggregate; and the development set shares no root with the 2026-10-01/10-07
development suite (160 roots), the 2026-10-07 round suites (122 + 122), the
hard draw's development and training sets (209 and 256 roots) or the
2026-10-04 preflight aggregate. The same check runs on torch against the
cell's own 370-root aggregate and the 2026-10-01, 2026-10-07 and 2026-10-21
aggregates and suites once stage 0 is done (section 4).

**Collection.** `scripts/run_dagger_research.py` with the cell's collection
arguments on 21 round-1 roots (three per family), the Gemma student replaced
by a stand-in (the observable expert deviating on 23% of its steps: a
redundant WLS, an off-route measurement context, or a refused phasor request;
`preflight_A/collect_preflight.py`): the report passes, 21 of 21 episodes,
307 labelled states, 170 safe rows (label yield 0.79), no export failure, no
unverifiable expert label, a 304-row mixture of `scada_only` rows with the 13
conventional schemas only. The quarantine reasons are the pipeline's usual
ones (meter and parameter corrections with several admissible targets fail
the observable rank-one proof, as 101 of 105 did in the 2026-10-07 cell's
round 1; rollbacks of a truth-safe candidate, which the stand-in's deviations
provoke, fail the offline audit).

**Expert end to end on the development draw.** `research/classifier_triage/expert_e2e.py
--profile scada_only --expert baseline_compare` on the 192 roots
(`preflight_A/e2e_development/`, 4.6 min):

| Family | N | Solved | Mean actions | Mean WLS calls | Comparisons |
| --- | --- | --- | --- | --- | --- |
| Single bad meter | 32 | 32 | 7.0 | 2.0 | 0 |
| Two to five bad meters | 24 | 23 | 23.0 | 4.8 | 0 |
| Line R/X error | 32 | 32 | 7.3 | 2.0 | 11 |
| Breaker status error | 32 | 32 | 8.0 | 2.0 | 0 |
| Meter + R/X error | 32 | 32 | 11.2 | 3.0 | 7 |
| Meter + breaker status | 24 | 24 | 12.0 | 3.0 | 0 |
| Clean controls | 16 | 16 | 2.0 | 1.0 | 0 |
| All | 192 | 191 | | | 18 |

No phasor or spectra request (none is on the surface), no triage report on
any solve, no invalid-action, false-commit or loop episode. The teacher
solves every parameter-bearing root, the misranked one included, after 18
comparisons. Its one failure is a multi-meter root with five bad voltage
magnitudes (r0_19daa0bb43de, buses 4, 8, 10, 11 and 14): two meter commits,
then no supported correction is left and it hands off after 17 actions. The 7
bus-split episodes are solved but carry `healthy_components_preserved=False`,
as 4 topology episodes of the classifier-profile run of Table I's teacher did
(`output/classifier_triage_20261004/e2e/classifier_baseline_v2`), all with a
truth-audited success; the flag belongs to how the evaluator's
healthy-component check reads a corrected bus split, not to this profile, and
the false-commit count is zero. So the teacher's ceiling on the fresh draw is
191 of 192 (99.5%, against 95 of 96 on Table I's roots).

## 4. Launch

Deployed 2026-10-22T06:32Z from 01ac598 (the incremental bundle 906374d..01ac598,
22.8 kB, uploaded over ssh stdin into a clone of the 2026-10-21 comparison
cell's source; the dry-run prerequisites passed):

```bash
PIPE=/scratch/yx3882/research_full_pipeline_20261022_conventional \
SOURCE_CELL=/scratch/yx3882/research_full_pipeline_20261021_compare BASE=906374d \
BRANCH=claude/clever-roentgen-03b19c OVERRIDES=conventional_only_20261022.env ALLOW_PARALLEL_CHAIN=1 \
bash research/hpc/full_pipeline_20260907/deploy_cell_from_windows.sh 01ac5983a1933eb117ad1685821618cbd7a2039f
```

Chain submitted 06:32:55 UTC: d0 20594198 (CPU partition `cs`, started at
once on cs623), bc0 20594219, r1c 20594220, r1t 20594222, r1e 20594224, r2c
20594225, r2t 20594226, r2e 20594227 (GPU stages on the
`a100|h100|h200|rtx6000` union with preemption opt-in and requeue, no GPU
type pinned). Status:

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20261022_conventional/status_pipeline.sh
```

GPU quota. The 2026-10-21 comparison cell (`research_full_pipeline_20261021_compare`)
runs at the same time: its r1t 20556243 holds one GPU (preempted and requeued
every 30 minutes or so) and its r1e 20556247 follows. The per-user limit of
QOS `gpu48` is 16 GPUs, so its one GPU does not by itself block this chain,
but on 2026-10-21 the cluster held new GPU allocations under
`QOSMaxGRESPerUser` for hours; this chain's GPU stages may queue behind that
cell or the same throttle. Both cells share the `fp-` job names, which is why
the submission needed `ALLOW_PARALLEL_CHAIN=1`; read the job ids above, not
the names.

Expected durations from the 2026-10-07 cell, scaled to this cell's sizes: d0
about 2 h for 370 roots, BC0 about 2.5 h (about 2,600 training rows, one
pass), each collection under 1 h for 74 roots, each round's training about 1
h, the round-1 evaluation (BC0, R1 and the expert on 192 roots) about 2.5 h
and round 2 (R1, R2) about 1.5 h, so a day or more with queueing.

**Stage 0 (done 07:57 UTC).** d0 20594198 ran 1 h 15 min on cs623: the
expert aggregate holds 3,648 rows from all 370 planned roots (train 2,747
rows from 279 roots, validation 533 from 54, test 368), none quarantined, no
unverifiable expert label; then the suite. BC0 20594219 started at 07:58 on
ga034 (A100) without queueing.

The suite on torch, seed 20261022 (`out/suite/`, `development.json` sha256
`9baf9ff6db36...`): 192 development roots (16/32/24/32/32/32/24) and 74 + 74
round roots. Development strata: parameter 18 dominant, 11 ambiguous, 3
misranked; meter+parameter 26 dominant, 6 ambiguous. This is the test set.
It shares 139 of its 192 scenario ids with the local preflight draw: every
topology, meter+topology, meter+parameter and clean root, and 13 of 32
single-meter, 8 of 24 multi-meter and 14 of 32 parameter roots. Seven of the
53 local-only roots are roots of the cluster's 370-root aggregate, which the
cluster draw therefore had to exclude; the others follow from the allocation
downstream of those exclusions and of platform-level numerical differences
(shared topology roots differ by at most 5e-11 pu between the Windows and
Linux solves, enough to change a fingerprint); I did not trace each one.

Disjointness on torch (`out/disjointness_check.json`, `check_suite_exclusion.py`,
passed): no physical root shared between the development set and the cell's
370-root aggregate or either round, none between the rounds, and none between
the development set and any of the eleven references: the aggregates of the
2026-10-01 ranked, 2026-10-07 classifier and 2026-10-21 comparison cells and
their suites (development 160, 160 and 209 roots; round training 122 + 122,
122 + 122 and 256).

The expert on the test set, before the cell's own expert arm: the same
end-to-end run as section 3 on the cluster's `development.json`
(`output/conventional_only_20261022/cluster_suite/e2e_development/` in the
worktree, 4.6 min) solves **192 of 192** (single meter 32, multi-meter 24,
R/X 32, breaker status 32, meter + R/X 32, meter + breaker 24, clean 16):
no phasor or spectra request, no false commit, no invalid-action or loop
episode, 25 comparisons (18 on R/X roots, 7 on meter + R/X roots), every
misranked root solved; mean actions 7.0, 20.6, 7.6, 8.0, 11.2, 12.0, 2.0 and
mean WLS calls 2.0, 4.3, 2.0, 2.0, 3.0, 3.0, 1.0 in the same order. The 7 bus
splits again carry the healthy-component flag with a truth-audited success.
The round-1 evaluation's expert arm rolls the same teacher out on the same
roots on torch and is the number the table uses.

Resume after a failure (receipts `out/d0.done`, `out/suite.done`,
`out/bc0.done`, `out/r{1,2}/collection.done`, `out/r{1,2}/training.done`,
`out/r{1,2}/round_summary.json`):

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch "cd /scratch/yx3882/research_full_pipeline_20261022_conventional && FROM=<stage> ALLOW_PARALLEL_CHAIN=1 bash submit_pipeline.sh"
```

## 5. Results

**BC0 and the round-1 collection (2026-10-22).** BC0 20594219 started 07:56
UTC on ga034 (A100), was preempted and restarted from the beginning at 08:38
on gr101 (RTX PRO 6000), and finished at 12:45: 687 optimizer steps over the
2,747-row training view, best validation loss 0.0046 at step 256 (that
checkpoint is the adapter). The round-1 collection 20594220 (gr101, 12:48 to
13:24) rolled BC0 out on the 74 round-1 roots: 74 of 74 episodes, 743
labelled states, 437 safe rows (label yield 0.867, against 0.864 in the
2026-10-07 cell), no invalid student action, student and teacher choosing a
different tool on 3.2% of the student's steps, no unverifiable expert label;
the mixture is 874 rows (437 collected and 437 drawn from the 2,747 D0 rows).
The safe targets include 13 breaker corrections and one comparison. The
round-1 training 20594222 (219 steps) started at 13:25 on ga007 (A100).
The torch SSH session expired at 12:29 UTC and was reopened by the user at
about 13:30; the chain ran on unaffected.

**Round 1 (2026-10-22, `out/r1/round_summary.json`, written 19:00 UTC).** R1
trained 219 steps on the 874-row mixture; the evaluation rolled BC0, R1 and
the teacher out on the 192 test roots (all 192 paired):

| Test roots (192) | Expert | BC0 | R1 |
| --- | --- | --- | --- |
| Successes | 192 | 189 | 191 |
| Invalid-action episodes (invalid actions) | 0 (0) | 19 (39) | 0 (0) |
| False-commit episodes | 0 | 3 | 0 |
| Loop episodes | 0 | 2 | 0 |
| Mean actions per episode | 9.87 | 10.38 | 9.85 |
| Mean WLS calls per episode | 2.50 | 2.60 | 2.50 |
| Bounded comparisons | 25 | 22 | 20 |
| Misranked parameter roots (3) | 3 | 3 | 3 |
| Ambiguous parameter roots (17) | 17 | 17 | 17 |

Every policy solves the three misranked roots the same way: one comparison,
then the comparison's best line, 8 actions. BC0's failures: a meter+R/X root
(r0_3386c1ad8700) where it cycles parameter correction, commit and rollback
into the 40-action budget (11 invalid actions, 8 false commits); a
meter+breaker root (r0_c253469c06d3) where it repeats the context requests and
meter edits into the budget (a loop); and a multi-meter root
(r0_f5eefea5a905) where it hands off after two meter repairs in 11 actions.
R1 fails only that multi-meter root, the same way, and has no invalid action,
false commit or loop on any root. Round 2 started at 18:56 UTC (r2c 20594225
on gr103).

**Round 2 and the cell's end (2026-10-23, `out/r2/round_summary.json` and
`out/pipeline_summary.json`, written 02:25 UTC).** R1 collected on the 74
round-2 roots (18:56 to 19:34 UTC): 74 of 74 episodes, 743 labelled states,
432 safe rows (label yield 0.867), no invalid student action, a different tool
from the teacher's on 1.6% of the student's steps, no unverifiable expert
label; the mixture is 864 rows (432 collected, 432 from the 3,184-row pool of
D0 and round 1). R2 trained 216 steps (20:03 to 21:45). The round-2
evaluation (R1 against R2 on the 192 test roots, 2 workers per policy) was
preempted twice and finished on an H200 at 02:25. R1 scores exactly its
round-1 numbers in the round-2 evaluation, family by family
(`adapter_consistency_across_rounds: consistent`). Preemptions over the whole
chain: BC0 once (restarted from the start), R1's training three times (resumed
from checkpoint 96), R2's evaluation twice; nothing failed.

**Per-policy table (the confirmation result).** The 192 freshly drawn test
roots, every policy on the same roots and measurement realizations, 40-action
budget, `scada_only` (no auxiliary tool, no triage report), every policy
trained on the seven conventional families only. Expert, BC0 and R1 from
`out/r1/round_summary.json`, R2 from `out/r2/round_summary.json`;
`research/hpc/full_pipeline_20260907/policy_table.py` (copies of both
summaries, the pipeline summary and the table in
`output/conventional_only_20261022/results/` of the worktree; sha256 of the
round summaries `0b1efc05...` and `3ff0d120...`).

| Cohort | N | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- | --- |
| Single bad meter | 32 | 32 | 32 | 32 | 32 |
| Two to five bad meters | 24 | 24 | 23 | 23 | 23 |
| Line R/X error | 32 | 32 | 32 | 32 | 32 |
| Breaker status error | 32 | 32 | 32 | 32 | 31 |
| Meter + R/X error | 32 | 32 | 31 | 32 | 32 |
| Meter + breaker status | 24 | 24 | 23 | 24 | 24 |
| Clean controls | 16 | 16 | 16 | 16 | 16 |
| **All test roots** | 192 | **192** | **189** | **191** | **190** |
| Episodes with an invalid action (invalid actions) | | 0 (0) | 19 (39) | 0 (0) | 9 (15) |
| Episodes with a false commit (false commits) | | 0 (0) | 3 (14) | 0 (0) | 0 (0) |
| Loop episodes | | 0 | 2 | 0 | 2 |
| Mean actions per episode | | 9.87 | 10.38 | 9.85 | 9.93 |
| Mean WLS calls per episode | | 2.50 | 2.60 | 2.50 | 2.51 |
| Bounded comparisons | | 25 | 22 | 20 | 21 |
| Misranked parameter roots (3) | | 3 | 3 | 3 | 3 |
| Ambiguous parameter roots (17) | | 17 | 17 | 17 | 17 |

WLS calls are the evaluator's per-episode `run_wls` plus `verify_candidate`
calls; the comparison's internal refits are not in them (the comparisons row
counts the calls). Per family, episodes with an invalid action / with a false
commit / with a loop, and mean actions and WLS calls:

| Cohort | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| Single bad meter | 0/0/0, 7.0, 2.00 | 0/0/0, 7.0, 2.00 | 0/0/0, 7.0, 2.00 | 0/0/0, 7.0, 2.00 |
| Two to five bad meters | 0/0/0, 20.6, 4.33 | 3/0/0, 20.5, 4.33 | 0/0/0, 20.7, 4.33 | 4/0/0, 20.9, 4.33 |
| Line R/X error | 0/0/0, 7.6, 2.00 | 0/0/0, 7.4, 2.00 | 0/0/0, 7.4, 2.00 | 0/0/0, 7.4, 2.00 |
| Breaker status error | 0/0/0, 8.0, 2.00 | 0/0/0, 8.2, 2.00 | 0/0/0, 8.0, 2.00 | 1/0/1, 7.9, 1.97 |
| Meter + R/X error | 0/0/0, 11.2, 3.00 | 7/2/0, 13.0, 3.38 | 0/0/0, 11.2, 3.00 | 1/0/0, 11.3, 3.03 |
| Meter + breaker status | 0/0/0, 12.0, 3.00 | 9/1/2, 13.7, 3.29 | 0/0/0, 12.0, 3.00 | 3/0/1, 12.3, 3.04 |
| Clean controls | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 |

R2 in detail. Its two failures: the multi-meter root every student misses
(r0_f5eefea5a905, handoff after two meter repairs in 11 actions) and an
isolated-terminal breaker root (r0_b7a5d222a6fe) where, with its breaker
candidate open, it requests a measurement context and then `correct_topology`
again; both are refused and the episode stops after 6 actions, non-terminal,
with a loop detected. Its **9 episodes with an invalid action** (15
invalid actions in all): four multi-meter roots (r0_07c625097ff5,
r0_2021f9f90fa4, r0_6a050ffec868, r0_85fa9fdc6ab2, two each, all solved), three
meter + breaker roots (r0_292a72a8c05c and r0_dd6143eb45bf one each, a second
`correct_topology` before verifying the first; r0_c98700d89772 two, the second
loop episode; all three solved), one meter + R/X root (r0_9d4100f19719, one,
solved) and the failed breaker root (two). No false commit.

Against Table I (2026-10-07 classifier cell, 96 conventional development
roots: expert 95, BC0 94, R1 95, R2 95; invalid-action episodes 0/2/3/1;
false-commit episodes 0/1/1/1; loops 0/1/0/0; mean actions
9.57/9.55/9.61/9.53): with the triage report and the auxiliary tools removed,
training on the conventional families only, a comparison-demonstrating
teacher, and a test set drawn fresh for this cell, the students again sit
within one or two roots of the teacher: R1 191 and R2 190 of 192 (99.5% and
99.0%) against the teacher's 192, BC0 189 (98.4%). Round 1 removes BC0's
invalid actions, false commits and loops entirely (19, 3 and 2 episodes to
none) at the teacher's episode length and solver cost; round 2 does not
improve on round 1 (one more lost root, 9 invalid-action and 2 loop episodes
where R1 had none, no false commit). The comparison step is learned from the
demonstrations: every policy, BC0 included, solves all three misranked
parameter roots with one comparison, where in the source cell no policy
solved its one misranked root. One multi-meter root (two meters repaired, the
rest abandoned) is missed by every student and solved by the teacher.

## 6. Follow-ups (2026-10-23): matched cloning control, stress test, round selection

Why. On the test set success is bounded by the ceiling (the teacher solves
192 of 192, BC0 189): BC0 against R1 is 2 discordant roots to 0, exact paired
p = 0.5. The round-1 change in process violations is not: episodes with an
invalid action go from 19 to 0 (19 discordant roots to 0, p = 3.8e-6), and
all 19 of BC0's fall on the 80 multi-error roots (meter + R/X, meter + breaker,
multiple meters; 11 to 21 teacher actions), none on the 112 single-error and
clean roots (2 to 8 actions; p = 1.5e-8 that all 19 land there by chance).
Three follow-ups test whether that change is the aggregation's doing and how
the iterates behave outside the training distribution. The design and the
selection rule below were written before any of these runs.

1. **Matched cloning control** (`control`). BC0 continued exactly as R1 was
   trained (BC0 as the initial adapter, 219 steps at 3e-5, micro-batch 1 x 4,
   seed 20260912, the same 128 validation rows) on R1's own 874-row mixture
   with each of its 437 learner-visited rows replaced, in place and within its
   family, by a row from the balanced training view of a fresh expert
   aggregate (the D0 settings; seed 20261123; plan 18/36/30/36/36/36/30, 222
   roots; any root of D0, the rounds, the test set, the validation set or the
   hard roots excluded). The control and R1 share the D0 rows, the row count,
   the family mix and every training setting; only the source of the new
   states differs: the expert's own distribution, the one BC0 was cloned
   from, against the student's. It is evaluated on the 192 test roots with
   R1's evaluation rows and seed, paired with BC0, R1 and the teacher.
   Reading fixed in advance: if the control keeps most of BC0's process
   violations (episodes with an invalid action, a false commit or a loop)
   while R1 has none, the round-1 change belongs to the on-policy states; if
   the control removes them too, it was more training on expert-distribution
   data. The seed: a local trial with 20261023 (the suite seed + 1, the test
   set's generator stream) reproduced 3 of 32 test roots, so the control uses
   an unused seed; with it 1 of 32 trial roots still equals a test root (a
   single-meter root drawn from the same tabular corpus row), which the
   exclusion drops.
2. **Stress test** (`hard`). BC0, R1, R2, the control and the teacher on the
   conventional roots of the 2026-10-15 hard draw's development set (193
   roots: 24 single-meter; 72 multi-meter, 48 of them flow-meter pairs that
   mimic a fault on one line; 41 R/X, of them 8 dominant, 24 ambiguous and 9
   misranked; 32 meter + R/X, 8, 16 and 8; 8 breaker, 8 meter + breaker, 8
   clean), less any root of this cell's D0, rounds or test set. Outside the
   training distribution: physical admission instead of the teacher-solvable
   one, meter errors drawn by WLS effect ratio (1.5 to 3) instead of the
   10-sigma floor, parameter rankings drawn by stratum.
3. **Round selection** (`validation`). BC0, R1 and R2 (with the control and
   the teacher) on the 96 conventional roots of the 2026-10-07 classifier
   cell's development suite (Table I's roots; disjoint from this test set),
   less any root of this cell's D0 or rounds. Rule, fixed before the run:
   the iterate with the most successes; ties go to fewer episodes with any
   process violation, then to fewer mean actions. The selected iterate's
   test-set row is the one to report as the DAgger result.

Code (`research/hpc/full_pipeline_20260907/`): `build_eval_suites.py`,
`build_control_mixture.py`, `evaluate_suite.py` (the runner's evaluation core
on any saved suite and adapters, no collection), `stage_extra_cpu.sbatch`,
`stage_control_train.sbatch`, `stage_eval_suite.sbatch`,
`submit_extras_20261023.sh`. Local smoke tests: the suite builder on the real
sources (96 validation and 193 hard roots), the evaluation entry point with
stand-in students and the teacher on seven validation roots, and the mixture
builder on the local collection preflight with a 32-root fresh aggregate
(families kept position by position, every replaced row expert-visited, the
13 conventional schemas only).

Deployed to the same cell from 331997f on 2026-10-23T18:06Z (no chain
resubmitted; the follow-ups write under `out/extra/`). Submitted 18:08 UTC
(`submit_extras_20261023.sh`): fp-xcpu 20748396 (suites, fresh aggregate,
control mixture); fp-ctrain 20748398 (after the CPU job); fp-xeval
hard_bc0_r1 20748400 and validation_bc0_r1 20748401 (after the CPU job),
test_control 20748402, hard_r2_control 20748403 and validation_r2_control
20748404 (after the control's training). Results:
`out/extra/evaluations/<name>/summary.json` and `episodes.jsonl`.

**The CPU stage (fp-xcpu 20748396, cs641, 2 h 18 min).** The suites keep
every candidate root (validation 96, hard 193; none is a root of this cell's
370-root D0, its rounds or its test set). The fresh expert aggregate (seed
20261123, 222 roots) holds 2,252 natural expert rows (its 666 counterfactual
recovery rows, 3 per root, are in the auxiliary file and enter no training
view; corrected 2026-10-27, section 7.1), none quarantined, no unverifiable
expert label; its balanced training view has
1,586 rows, 214 of which belong to roots excluded from the control (D0, round,
test, validation or hard roots). The control mixture
(`out/extra/control/mixture.jsonl`, sha256 `ee7ffd3d...`) has R1's 874 rows,
the 437 D0 rows in place and 437 fresh rows in the places of the collected
ones at R1's family counts (measurement 47, meter + R/X 76, meter + breaker
73, multi-meter 126, clean 4, R/X 48, breaker 63; candidates per family 119
to 286, clean 52); none of its states was visited by the student.

The first control training job (fp-ctrain 20748398) stopped at once on a
wrong report path in `stage_control_train.sbatch` (it looked for
`mixture.jsonl.report.json`; the builder writes `mixture.report.json`), and
Slurm cancelled its three dependents (20748402 to 20748404). Fixed in
4d79282, redeployed at 21:47 UTC (only that script changed; the two running
evaluations were untouched) and resubmitted at 21:48: fp-ctrain 20762467,
then test_control 20762468, hard_r2_control 20762469 and
validation_r2_control 20762470. hard_bc0_r1 20748400 (gr103) and
validation_bc0_r1 20748401 (gr104) started at 21:35 and 21:38.

**Results (2026-10-24, all seven jobs completed by 01:56 UTC).** The control
trained 219 steps on gr102 (RTX PRO 6000, 1 h 47 min); the five evaluations
ran 59 min to 2 h 05 min (hard_bc0_r1 was preempted twice and resumed from its
episode checkpoints). Tables from `followup_tables.py` (`out/extra/followup_tables.{json,md}`
on torch; copies in `output/conventional_only_20261022/results/` of the
worktree). Process-violation episodes are episodes with an invalid action, a
false commit or a loop; p values are exact two-sided McNemar tests on the
paired roots against R1.

Test set (the 192 roots; expert, BC0, R1 and R2 as in section 5):

| | Expert | BC0 | R1 | R2 | Control |
| --- | --- | --- | --- | --- | --- |
| Single bad meter (32) | 32 | 32 | 32 | 32 | 32 |
| Two to five bad meters (24) | 24 | 23 | 23 | 23 | 23 |
| Line R/X error (32) | 32 | 32 | 32 | 32 | 29 |
| Breaker status error (32) | 32 | 32 | 32 | 31 | 32 |
| Meter + R/X error (32) | 32 | 31 | 32 | 32 | 27 |
| Meter + breaker status (24) | 24 | 23 | 24 | 24 | 24 |
| Clean controls (16) | 16 | 16 | 16 | 16 | 16 |
| **All (192)** | **192** | **189** | **191** | **190** | **183** |
| Misranked parameter roots (3) | 3 | 3 | 3 | 3 | 0 |
| Process-violation episodes | 0 | 19 | 0 | 9 | 8 |
| (invalid action / false commit / loop) | 0/0/0 | 19/3/2 | 0/0/0 | 9/0/2 | 8/7/2 |
| Mean actions; mean WLS calls | 9.87; 2.50 | 10.38; 2.60 | 9.85; 2.50 | 9.93; 2.51 | 10.96; 2.82 |
| Bounded comparisons | 25 | 22 | 20 | 21 | 1 |
| Failures only this / only R1 (p) | 0/1 (1) | 2/0 (0.5) | | 1/0 (1) | 8/0 (0.008) |
| Violations only this / only R1 (p) | 0/0 | 19/0 (4e-6) | | 9/0 (0.004) | 8/0 (0.008) |

Hard-draw stress roots (193, outside the training distribution):

| | Expert | BC0 | R1 | R2 | Control |
| --- | --- | --- | --- | --- | --- |
| **All (193)** | **185** | **182** | **183** | **182** | **170** |
| R/X error (41) / meter + R/X (32) | 41 / 32 | 40 / 30 | 40 / 31 | 39 / 31 | 33 / 25 |
| Misranked (17) / ambiguous (40) / dominant (16) | 17/40/16 | 15/40/15 | 15/40/16 | 14/40/16 | 2/40/16 |
| Multi-meter (72; 48 meter-pair mimics) | 64 | 64 | 64 | 64 | 64 |
| Process-violation episodes | 0 | 22 | 1 | 5 | 9 |
| (invalid action / false commit / loop) | 0/0/0 | 22/2/1 | 1/1/1 | 5/2/0 | 8/6/2 |
| Mean actions; mean WLS calls | 11.30; 2.70 | 11.39; 2.74 | 11.31; 2.73 | 11.42; 2.76 | 11.80; 2.92 |
| Bounded comparisons | 60 | 57 | 55 | 54 | 3 |
| Failures only this / only R1 (p) | 0/2 (0.5) | 1/0 (1) | | 1/0 (1) | 13/0 (2e-4) |
| Violations only this / only R1 (p) | 0/1 (1) | 21/0 (1e-6) | | 4/0 (0.13) | 8/0 (0.008) |

Every other family is solved by every policy (single meter 24, breaker 8,
meter + breaker 8, clean 8); the 8 multi-meter roots no policy solves are
meter-pair mimics (20 of 24 of each sign for every policy, the teacher
included).

Validation roots (96) and the round selection:

| | Expert | BC0 | R1 | R2 | Control |
| --- | --- | --- | --- | --- | --- |
| **All (96)** | **96** | **96** | **96** | **96** | **92** |
| Process-violation episodes | 0 | 6 | 1 | 4 | 8 |
| (invalid action / false commit / loop) | 0/0/0 | 6/0/0 | 1/0/0 | 4/0/1 | 7/5/1 |
| Mean actions; mean WLS calls | 10.01; 2.56 | 10.07; 2.56 | 10.05; 2.56 | 10.07; 2.57 | 11.59; 2.99 |
| Violations only this / only R1 (p) | 0/1 (1) | 5/0 (0.06) | | 4/1 (0.38) | 8/1 (0.04) |

The control's four misses are three meter + R/X roots and the misranked R/X
root. **Round selection (rule of section 6): BC0, R1 and R2 tie at 96
successes; R1 has the fewest process-violation episodes (1, against R2's 4 and
BC0's 6). Selected: R1**, whose test row is 191 of 192 with no
process-violation episode, at the teacher's episode length (9.85 actions
against 9.87) and solver cost (2.50 WLS calls against 2.50).

Why the control went the other way. The two mixtures share the 437 D0 rows,
the families and every training setting; the other 437 rows teach different
things. R1's collected rows, at states the student reached, are mostly which
context to request (190: measurement 104, parameter 55, breaker 31), commits
(88), solves (88) and handoffs (47), with 19 corrections (6 meter, 13
breaker, no parameter correction: the observable rank-one proof keeps few
correction labels at student states) and 1 comparison. The fresh expert rows
hold 110 corrections (69 meter, 23 parameter, 18 breaker) against 4
comparisons. Continued cloning on them moved the policy toward correcting the
top-ranked line directly: the control compares on 1 test root against R1's
20, fails all three misranked test roots and 15 of 17 on the hard draw, and
commits wrong candidates (7 false-commit episodes on the test set). Its loss
on the 128 held-out D0 rows rose over the pass (0.0086 at step 16 to 0.0100
at step 219, lowest 0.0076 at step 32), while R1's fell (0.0090 to 0.0011,
lowest 0.00065 at step 144).

Reading. (1) The round-1 change is not explained by more training: with the
same rows, steps and optimizer on expert-distribution data, the control still
has 8 process-violation episodes on the test set (BC0 19, R1 0) and 9 on the
hard draw (BC0 22, R1 1), and it loses success (183 against BC0's 189 and
R1's 191), whereas the on-policy round removes them (19 to 0 on the test set,
22 to 1 on the hard draw, 6 to 1 on validation; paired p 4e-6 and 1e-6 on the
first two). (2) The reliability gain holds out of distribution: on the
hard draw all iterates stay within 2 to 3 roots of the teacher (182 to 183
against 185; the teacher solves all 17 misranked roots, R1 15), and BC0's
violation episodes persist (22) where R1 has 1. (3) Round 2 adds nothing on
any set (test 9 violation episodes against R1's 0, p = 0.004; hard 5 against
1; validation 4 against 1), and the validation set selects R1. One control,
one seed: the control's degradation may partly reflect its row mix (more
direct corrections than the collected rows hold), which is what
expert-distribution data looks like in this pipeline; repeated seeds would
bound it.

## 7. Follow-ups (2026-10-27): what the cloning baseline trained on, and success under a fixed action budget

Why. The paper session (PES GM 2027, `docs/papers/pesgm2027`) finds the
success gap between BC0 and the DAgger iterates on the test set (189 against
191 and 190 of 192) too small to carry the manuscript's claim and asked for
two additions on metrics fixed before the runs: a pure-cloning baseline
trained on the natural expert trajectories only, on the reading that the
aggregate's injected-recovery demonstrations (`D0_COUNTERFACTUALS=3`; "666 of
2,252 rows" in section 6) make BC0 a perturbation-augmented clone that has
already seen off-expert states; and the success-under-budget curve at 12, 16
and 20 actions measured by re-running the policies under those horizons
instead of being read off the 40-action episode lengths (that reading gave
expert 169, BC0 157 and R1 168 of 192 at 12 actions). Branch
`claude/competent-haibt-e7b0ad`, fast-forwarded from
`claude/clever-roentgen-03b19c` at d6e6acc; the cell is the same, the records
go under `out/extra/`.

### 7.1 The cloning baseline's training data (verified 2026-10-27, before any run)

The premise of the first addition does not hold: no injected-recovery row is
in any training view of this cell, so the "BC-natural" adapter as specified
is BC0 itself. The counts, read from the cell's files on torch:

| File (`out/d0/`) | Rows | Distinct ids | Roots | `dataset_source` | `state_visited_by` | `production_label_eligible` |
| --- | --- | --- | --- | --- | --- | --- |
| `aggregate.raw.jsonl` (the expert's rollouts, beta 1) | 3,648 | 3,648 | 370 | dagger_rollout 3,648 | expert 3,278, initial 370 | True 3,648 |
| `aggregate.auxiliary_counterfactual.raw.jsonl` (the counterfactual branches) | 1,110 | 1,110 | 370 | synthetic_counterfactual 1,110 | counterfactual_injection 1,110 | False 1,110 |
| `aggregate.train_view.jsonl` (BC0's training file) | 2,747 | 2,171 | 279 | dagger_rollout 2,747 | expert 2,474, initial 273 | True 2,747 |
| `aggregate.validation.jsonl` (the 128 validation rows are drawn from it) | 533 | 533 | 54 | dagger_rollout 533 | expert 479, initial 54 | True 533 |

How this comes about (`psse_env/examples/generate_round0_aggregate.py`): the
round-0 collector rolls the expert out on every planned root (`collect_round0`,
beta 1; `executed_by: expert` on every row), then, for
`--counterfactuals-per-scenario 3`, the counterfactual generator executes three
injected learner mistakes per root in isolated branches and records the
expert's recovery. Those rows are stamped `dataset_source:
synthetic_counterfactual`, `state_visited_by: counterfactual_injection` and
`production_label_eligible: False` (their injected actions may be constructed
from hidden truth, `psse_env/dagger/counterfactual_generator.py`), and the
export splits every row by that flag: the eligible rows go to
`aggregate.raw.jsonl` and its train, validation and test splits, the
ineligible ones to the auxiliary file. The balanced training view is built
from the train split with `require_production_label_eligible=True`, so the
training view, the validation subset and BC0's training (`research_run.json`:
2,747 training rows; the file's sha256 is in `d0.done`) hold natural rows
only. `recovery_stratum`, `collection_role` and `research_label_eligible` are
null on every row of every view. The `state_class` of the training view is
clean_successful 1,073, accepted_partial_continuation 714,
accepted_final_commit 390, terminal_operator_escalation 275,
accepted_partial_commit 248, terminal_resolved 45 and
rejected_candidate_recovery 2 (the expert's own observable recovery on two
roots, not an injection). The count the chain printed, `recovery_rows: 1110`
(`d0_console.json`), is the auxiliary file.

What the balanced view does to the natural rows: the 279 training roots have
2,747 natural rows, and the view keeps 2,747 rows, but 576 of the natural rows
(300 accepted_partial_continuation, 258 clean_successful, 18
terminal_operator_escalation) are left out and 576 others appear twice
(`max_duplicate_count=2`; the sampler's state-class, tool-category, family and
per-root targets). None of the dropped rows is a low-margin row (`cost_margin`
at most 0.05: none in the aggregate).

Section 6's sentence on the control's fresh aggregate, "holds 2,252 rows (666
of them forced-recovery rows)", is corrected today: that aggregate holds 2,252
natural rows (`aggregate.raw.jsonl`, all `dagger_rollout`,
`production_label_eligible` True), its 666 counterfactual rows (222 roots x 3,
the same structure) are in its auxiliary file, and its 1,586-row balanced view
holds natural rows only, so the control, too, continued on expert-visited
states only. The sentence had read `control_d0_console.json` (`episode_rows:
2252, recovery_rows: 666`) as if the second count were part of the first.

Consequence. BC0 is the pure behavioral-cloning baseline the manuscript
needs: one pass over the balanced view of the expert's own trajectories, no
injected state, no recovery demonstration beyond what the expert's own
rollouts contain (the two rejected-candidate rows). The third explanation of
the small success gap, BC0 as perturbation-augmented cloning, is withdrawn;
the two others stand (112 single-error and clean roots solved in 2 to 8
actions where nothing compounds; the transactional controller makes BC0's
mistakes recoverable, 16 of its 19 violation episodes still succeed). No
"BC-natural" adapter is trained: filtering the view by the intervention fields
removes 0 of 2,747 rows, and retraining on identical rows with the same seed
and recipe would reproduce BC0 up to kernel nondeterminism. Whether a
different cloning variant is wanted instead (a repeat of BC0's training to
bound that nondeterminism; or cloning on the 2,747 natural rows each once,
without the balancing sampler, at BC0's 687 steps) was put to the paper
session on 2026-10-27. `submit_extras_20261027.sh` (`ADAPTER=label=/path`) and
`followup_tables.py --extra-policy label` evaluate any such adapter on the
three suites, paired with the existing records, with the safety-clean success
and the paired tests against BC0 and R1 that this section's tables use.

### 7.2 Budget-limited evaluation: design (written before the runs)

Question. The success-under-budget counts of the paper session's analysis
were read off the 40-action episodes (a root solved in at most b actions
counts as solved under b). That ignores that the students and the teacher act
on the budget: the environment binds its action limit to the horizon, every
observation carries `remaining_budget` counting down from it (the students see
it in their prompt, the observation is the user payload, as they did in
training from 40 down; `psse_env/dagger/evaluator.py`,
`psse_env/episode_budget.py`), and the teacher hands off before opening a
correction lifecycle it cannot close (`_recovery_budget_proposals` in
`psse_env/oracle/expert_policy.py`, fewer than 4 actions left) and withholds
bounded joint retries below 4 remaining actions. Here the policies are re-run
under the horizons.

Runs. `research/hpc/full_pipeline_20260907/submit_extras_20261027.sh`: three
`fp-xeval` jobs, `test_budget12`, `test_budget16` and `test_budget20`, each
`evaluate_suite.py` on the 192 test roots
(`out/r1/collection/development_scenarios.json`, the round-1 evaluation's rows
and measurement realizations) with BC0 (`out/bc0/lora`) and R1
(`out/r1/training/lora`, the iterate section 6 selected) in the two adapter
slots and the teacher (`baseline_compare`) beside them, seed `SEED_ROUND`
20260912 as in every evaluation of this cell, `scada_only`, the same
workers. The only change is `EVAL_MAX_STEPS` = 12, 16 or 20 in the job's spec,
which `stage_eval_suite.sbatch` passes as `--max-steps` (the spec is sourced
after `pipeline.env`, whose value is the cell's 40). The round-1 evaluation is
the 40-action point of the curve. Every evaluation records its horizon and
adapter tree hashes in `policies.json`; `budget_tables.py` refuses an
evaluation whose horizon or adapters differ from the budget and the cell's
receipts.

Metrics, fixed now. Per budget and policy on the 192 roots, and per family:
successes (truth-audited), safety-clean successes (success with no invalid
action, false commit or loop), episodes with an invalid action, a false commit
or a loop, episodes that ran to the budget, mean actions and mean WLS calls.
Paired exact two-sided McNemar tests at each budget between BC0 and R1 and of
each against the teacher, on success, on safety-clean success and on process
violations. The 40-action approximation (solved within b actions at 40) is
reported next to the measured count with the roots on which the two disagree,
for every policy; the measured count is the result.

Reading, fixed now. The budget curve is a statement about episode length: at
12 actions the multi-meter roots the teacher needs about 20 actions for are
out of reach for every policy, so the curve separates the policies only where
their episode lengths differ, chiefly on the multi-error roots (BC0 spends
1.7 more actions there). More successes for R1 than for BC0 at a budget are a
difference in efficiency, not in what the policies solve at 40; the
safety-clean count at each budget adds the process dimension.

Expected cost: about 1 to 2 GPU-hours per budget (the 40-action evaluation of
the three arms took 2.5 h; the budgets cap BC0's longest episodes).

**Launch.** Deployed 2026-10-27T05:49Z from efbc953 (the incremental bundle
4d79282..efbc953, 24.5 kB, uploaded over ssh stdin into the cell's source,
which had stood at 4d79282; `pipeline.overrides.env` unchanged; the dry-run
prerequisites passed). Submitted 05:50 UTC with `submit_extras_20261027.sh`
(no chain resubmitted): fp-xeval test_budget12 21070126, test_budget16
21070127 and test_budget20 21070128, on the `a100|h100|h200|rtx6000` union
with preemption opt-in and requeue, no GPU type pinned, 2 workers per policy
on a GPU of 70 GB or more. Specs `out/extra/specs/test_budget{12,16,20}.env`
(`EVAL_MAX_STEPS` 12, 16, 20; `EVAL_EXPERT=1`); records
`out/extra/evaluations/test_budget<b>/` (`policies.json` with the horizon and
the adapter trees, `{expert,bc0,r1}_eval.json`, `episodes.jsonl`,
`summary.json`); tables from
`research/hpc/full_pipeline_20260907/budget_tables.py --cell-dir $PIPE`.
Status:

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch "export PATH=/opt/slurm/bin:\$PATH; sacct -j 21070126,21070127,21070128 -o JobID%12,JobName%10,State%12,Elapsed%12,NodeList%10; ls /scratch/yx3882/research_full_pipeline_20261022_conventional/out/extra/evaluations/"
```
