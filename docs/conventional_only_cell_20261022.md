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
session on 2026-10-27. **Decision (paper session, 2026-10-27): no substitute
arm.** BC0 stays the pure cloning baseline and the manuscript cites this
section; the sampler ablation answers a different question, and a repeat of
the training would only bound kernel nondeterminism around a comparison the
paper no longer leans on for success (its paired tests are on the violation
episodes, 19 against 0). The one optional extra is a repeat of BC0's training
under another seed, reported as the seed stability of its 189 successes and
19 violation episodes rather than as an arm, to be run only after the
budget-limited evaluations and only on idle GPU capacity.
`submit_extras_20261027.sh` (`ADAPTER=label=/path`) and
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
The three jobs completed by 07:58 UTC (12 actions: 1 h 34 min on gr102; 16 and
20 actions: preempted once at about 06:15 UTC, resumed from the per-episode
checkpoints on gr104 and gh120, 1 h 25 min and 1 h 21 min of run time).

**R2 at the budgets** (asked by the paper session on 2026-10-27, since Table I
carries an R2 column): deployed 13:12 UTC from 4d34050 (the submit script's
`ADAPTER_BUDGETS=1`, the R2-aware `budget_tables.py`), submitted 13:15 UTC:
fp-xeval test_budget12_r2 21101407, test_budget16_r2 21101408 and
test_budget20_r2 21101409, each R2 alone (`out/r2/training/lora`, the
evaluator's first adapter slot, no expert arm) on the same rows, seed and
horizon as the matching three-arm job; records
`out/extra/evaluations/test_budget<b>_r2/`. R2's 40-action point is the
candidate slot of the round-2 evaluation. The three jobs ran 51 to 55 min
each on gr102 and completed by 14:13 UTC.
Status:

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch "/opt/slurm/bin/sacct -j 21070126,21070127,21070128,21101407,21101408,21101409 -o JobID%12,JobName%10,State%12,Elapsed%12,NodeList%10; ls /scratch/yx3882/research_full_pipeline_20261022_conventional/out/extra/evaluations/"
```

**Results (2026-10-27; `out/extra/budget_tables.{json,md}` on torch, copies
with the per-episode records, specs and console receipts under
`output/conventional_only_20261022/results/budget_20261027/` of the
worktree).** Every policy on the 192 test roots and measurement realizations,
seed 20260912; the 40-action row is the round-1 evaluation (R2: the round-2
evaluation's candidate slot); every budget-limited evaluation's
`policies.json` records its horizon and the tree hashes of `bc0.done`,
`r1/training.done` and `r2/training.done`. Successes, with safety-clean
successes (solved with no invalid action, false commit or loop) in
parentheses:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 177 (177) | 172 (161) | 176 (176) | 175 (171) |
| 16 | 178 (178) | 174 (164) | 177 (177) | 177 (172) |
| 20 | 181 (181) | 179 (165) | 180 (180) | 179 (175) |
| 40 | 192 (192) | 189 (172) | 191 (191) | 190 (182) |

Process-violation episodes (with an invalid action / a false commit / a
loop; a false commit is a commit requested for a candidate the verifier had
not accepted, which the gate refuses), then the episodes that ran to the
budget:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 0; 53 | 13 (13/1/1); 57 | 0; 53 | 5 (5/1/1); 55 |
| 16 | 0; 1 | 13 (13/3/1); 19 | 0; 14 | 6 (6/1/1); 15 |
| 20 | 0; 0 | 16 (16/3/1); 14 | 0; 11 | 7 (7/0/2); 11 |
| 40 | 0; 0 | 19 (19/3/2); 2 | 0; 0 | 9 (9/0/2); 0 |

The 80 multi-error roots (two to five meters, meter + R/X, meter + breaker):
successes (safety-clean); violation episodes:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 65 (65); 0 | 60 (50); 12 | 64 (64); 0 | 63 (60); 4 |
| 16 | 66 (66); 0 | 62 (52); 13 | 65 (65); 0 | 65 (61); 5 |
| 20 | 69 (69); 0 | 67 (53); 16 | 68 (68); 0 | 68 (64); 6 |
| 40 | 80 (80); 0 | 77 (60); 19 | 79 (79); 0 | 79 (71); 8 |

The 112 single-error and clean roots are solved by every policy at every
budget, except that R2 fails the isolated-terminal breaker root of section 5
at 20 and 40 actions (111); safety-clean except one BC0 breaker-status root
at 12 actions and one R2 root per budget (an R/X root at 12 and 16, that
breaker root at 20 and 40). Per family, the roots not solved are multi-meter
roots for every policy (of 24: expert 9, 10 and 13 at 12, 16 and 20 actions;
BC0 7, 10, 14; R1 8, 9, 12; R2 8, 9, 12), for BC0 meter + R/X roots (31, 29,
30 of 32) and meter + breaker roots (22, 23, 23 of 24), and for R2 one meter +
breaker root at 12 actions; every other family is 100% at every budget. Mean
actions and mean WLS calls per episode:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 8.79; 2.33 | 8.84; 2.34 | 8.77; 2.34 | 8.78; 2.35 |
| 16 | 9.01; 2.34 | 9.38; 2.44 | 9.11; 2.41 | 9.18; 2.42 |
| 20 | 9.35; 2.41 | 9.73; 2.48 | 9.39; 2.41 | 9.43; 2.42 |
| 40 | 9.87; 2.50 | 10.38; 2.60 | 9.85; 2.50 | 9.93; 2.51 |

Paired exact two-sided McNemar tests (discordant roots, first policy /
second policy; p):

| Budget | BC0 vs R1, success | BC0 vs R1, safety-clean | BC0 vs expert, success | BC0 vs expert, safety-clean | R1 vs expert, success and safety-clean | R2 vs R1, success | R2 vs R1, safety-clean | R2 vs expert, safety-clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 12 | 4/0 (0.125) | 15/0 (6.1e-5) | 5/0 (0.0625) | 16/0 (3.1e-5) | 1/0 (1) | 1/0 (1) | 5/0 (0.0625) | 6/0 (0.031) |
| 16 | 4/1 (0.375) | 14/1 (9.8e-4) | 5/1 (0.219) | 15/1 (5.2e-4) | 1/0 (1) | 0/0 (1) | 5/0 (0.0625) | 6/0 (0.031) |
| 20 | 3/2 (1) | 16/1 (2.7e-4) | 4/2 (0.688) | 17/1 (1.4e-4) | 1/0 (1) | 1/0 (1) | 5/0 (0.0625) | 6/0 (0.031) |
| 40 | 2/0 (0.5) | 19/0 (3.8e-6) | 3/0 (0.25) | 20/0 (1.9e-6) | 1/0 (1) | 1/0 (1) | 9/0 (0.0039) | 10/0 (0.002) |

BC0 against R2 on safety-clean success: 12/2 (p = 0.013), 11/3 (0.057),
13/3 (0.021) and 15/5 (0.041) at 12, 16, 20 and 40 actions.

Against the approximation from the 40-action episodes (a root solved in at
most b actions at 40 counts as solved under b), the reading the manuscript
had printed as a lower bound: at 12 actions the measured counts are higher
for every policy, expert 169 to 177, BC0 157 to 172, R1 168 to 176, R2 162 to
175, and no root counted by the approximation is lost in the measured run
(safety-clean: BC0 152 to 161 with 11 roots only measured and 2 only
approximated, R2 162 to 171 with 10 and 1); at 16 actions the totals
coincide, 178, 174, 177 and 176 to 177 (BC0 trades one root each way;
safety-clean BC0 161 to 164, R2 172 with 2 each way); at 20 actions 181, 177
to 179, 180 and 179 (safety-clean BC0 164 to 165 with four roots each way, R2
175 with one each way). The 12-action gap is the budget in the observation:
the teacher hands off instead of opening a lifecycle it cannot close (53 of
its episodes end at the budget), and the students shorten their episodes the
same way (mean actions 8.8 at 12 actions against 9.9 to 10.4 at 40 for every
policy; 53 to 57 episodes at the budget).

Reading. (1) Under every budget the success gap between BC0 and R1 is 4 to 5
roots (172 against 176 at 12 actions, 174 against 177, 179 against 180) and
not significant (p 0.125 to 1); the same holds against the teacher. (2) The
safety-clean gap is 11 to 15 roots at every budget and significant (p at most
1e-3): BC0 keeps 13, 13 and 16 process-violation episodes under the budgets,
all on multi-error roots but one, while R1 has none at any budget and equals
the teacher's safety-clean count minus the one multi-meter root every
student misses. (3) R1's episode length and solver cost are the teacher's at
every budget (8.77 against 8.79 actions at 12, 9.11 against 9.01 at 16, 9.39
against 9.35 at 20; 2.34 to 2.41 WLS calls); BC0 spends more at 16 and 20
actions (9.38 and 9.73) and half an action more at 40. (4) R2 sits between
the two: one success under R1 at 12 and 20 actions and equal at 16, but 5
safety-clean successes under R1 at every budget (p = 0.0625 each; 9 at 40, p
= 0.0039) and 6 under the teacher (p = 0.031), from 5 to 7 violation episodes
under the budgets; round 2 adds nothing to round 1 under any budget, as
section 6 found at 40 actions. (5) The budget curve is the multi-meter
curve: with two context requests and about six actions per meter, 12 actions
hold the two-meter roots only (the teacher solves 9 of 24), 20 actions about
half (13) and 40 all 24; outside that family the budget costs only BC0,
whose invalid actions and false commits spend the actions a meter + R/X or
meter + breaker root needs.

## 8. A second fresh test draw (2026-10-29): 192 more scenarios, pooled to 384

Why. The manuscript's Table I reads the 192-root test set of this cell. A
second draw of the same composition doubles the test set to 384 scenarios and
replicates every row on roots that no policy, draw or earlier cell has
touched (asked by the paper session on 2026-10-28 and approved by the user;
the results are wanted by 5 November). Branch `claude/competent-haibt-e7b0ad`,
the same cell; records under `out/extra/`. No file under `psse_env/` or
`scripts/` differs from the cell's recorded source 01ac598; the commits since
then change only `research/hpc/`, the tests and this note.

### 8.1 Design (written before the runs)

**The draw** (`stage_test_draw.sbatch`, receipt
`out/extra/test_draw2/draw.done`). Three steps:

1. `collect_reference_roots.py` gathers every physical root of every earlier
   set into `out/extra/test_draw2/protected_roots.json` (report beside it,
   per file: size, sha256, rows, roots, rows without a root): the expert
   aggregate and the development and round suites of every cell under
   `/scratch/yx3882/research_full_pipeline_*` (13 cells besides this one, the
   IEEE 57 and IEEE 118 cells included although their roots cannot coincide),
   this cell's aggregate, test set and round suites, the validation and hard
   suites of section 6 and their two sources, the control's aggregate, the
   2026-10-15 hard draw (development and training sets, both renderings) and
   the local preflight's development-only roots (section 4).
2. `build_suite.py` exactly as stage 0 drew the first suite (section 4): the
   cell's round and development plans, two rounds, candidate multiplier 3,
   training threshold 1.2, development threshold 1.0 with rank allowance 2,
   the 10-sigma meter floor, the dangling-terminal and bus-split effects,
   `recoverable` admission, `scada_only`, the same corpora and system, the
   cell's D0 as `--d0-raw`, the frozen BC0 suite and the trace validation set
   protected as in stage 0, and the collected roots protected on top, with
   seed 20261029 (development stream 20261030; the first draw used 20261022
   and 20261023, the control's aggregate 20261123). Output
   `out/extra/test_draw2/suite/`. The round training files the builder writes
   are a by-product of running the identical procedure and are not used.
   Rule fixed now: if a family runs short of fresh candidates after the
   exclusions (the builder stops with "Increase --candidate-multiplier"), the
   draw is repeated once with multiplier 6. The candidates are the same
   generator's identically distributed draws, so a larger pool changes which
   are kept, not their distribution; the receipt records the multiplier used.
3. `check_suite_exclusion.py`, the first draw's check: the new suites against
   the cell's D0 and each other, and the new development set against every
   reference file it can read and against the collected roots
   (`disjointness_check.json`); then `suite_stats.py` on both draws
   (`suite_stats.{json,md}`): meter errors per root, channels and magnitudes
   in pu and in sigma; R and X factors; ranking strata, dominance ratios and
   true-line ranks; topology effects, breakers, load scales and operating
   points; parameter scans and substation telemetry per root.

Composition, as in the first draw: 16 clean, 32 single-meter, 24 multi-meter,
32 R/X, 32 breaker-status, 32 meter + R/X and 24 meter + breaker roots; the
parameter strata are whatever the draw yields at the development threshold
(the first draw: 18 dominant, 11 ambiguous and 3 misranked R/X roots; 26
dominant and 6 ambiguous meter + R/X roots). The draw keeps the first draw's
family-specific metadata (parameter scans on the R/X families, substation
telemetry on the breaker families): the uniform-metadata convention decided
on 2026-10-28 applies to suites built for new studies, and this draw
replicates the conditions the policies were trained and tested under.

**The evaluations** (`submit_test_draw2_20261029.sh`; specs `test2_40`,
`test2_40_r2`, `test2_budget12`, `test2_budget12_r2`, `test2_budget20`,
`test2_budget20_r2` under `out/extra/specs/`): BC0, R1 and the teacher, and R2
alone, on the new draw at 40 actions after the draw, then at 12 and 20
actions after the 40-action jobs, each with `evaluate_suite.py` under the
cell's recorded environment options, model and teacher (`baseline_compare`,
`scada_only`), seed 20260912, and 2 workers per policy on a GPU of 70 GB or
more, as every evaluation of this cell. The adapters are the cell's
(`out/bc0/lora`, `out/r1/training/lora`, `out/r2/training/lora`), checked by
tree hash against the stage receipts. 16 actions is not in the manuscript's
table and is not run.

**Metrics, fixed now** (`pooled_tables.py`, output
`out/extra/pooled_tables.{json,md}` and `pooled_tables_episodes.jsonl`). For
the new draw alone and pooled with the first (384 roots; the draws are
root-disjoint, so the pool is the union), at 40, 12 and 20 actions: successes,
safety-clean successes (solved with no invalid action, false commit or loop; a
false commit is a commit requested for a candidate the verifier had not
accepted, which the gate refuses, so nothing is kept), violation episodes,
episodes that ran to the budget, mean actions and WLS calls, per family and
per error cohort (multi-error: multiple meters, meter + R/X, meter + breaker;
the rest single-error and clean); exact two-sided McNemar tests BC0 vs R1,
BC0 vs expert and R1 vs R2 (and the other pairs) on success, on safety-clean
success and on violations; per family and policy, draw 1 against draw 2 at 40
actions. Per-episode records in the schema of section 7 (`episodes.jsonl` in
every evaluation directory; the pooled file adds the draw and the horizon to
the same record).

**Reading, fixed now.** The pooled rows are sums over the two draws. A family
whose successes differ between the draws by more than two roots for the same
policy, or whose strata, error magnitudes or effects differ in the suite
statistics, is flagged for the manuscript to explain; nothing is redrawn or
dropped after the evaluations.

**Launch.** First submitted 2026-10-29T01:58Z from f005b4f (jobs 21304171 to
21304181) and cancelled three minutes in, while the builder was drawing:
the reference collection had read 8,080 roots from 60 files but none of the
53 in `local_only_dev_roots.json`, which stores them under the key `fp`
(fixed in 8e07f94; the collector now reads `physical_root`, `fp` and `root`
as well). Redeployed from 8e07f94 at 02:11 UTC (the incremental bundle
f005b4f..8e07f94 over ssh stdin; the dry-run prerequisites passed), the
cancelled run's partial outputs removed, and resubmitted at 02:12 UTC with
`submit_test_draw2_20261029.sh`: fp-xdraw 21305137 (CPU partition `cs`), then
fp-xeval test2_40 21305138 and test2_40_r2 21305139 (after the draw), then
test2_budget12 21305140 and test2_budget20 21305143 (after test2_40) and
test2_budget12_r2 21305142 and test2_budget20_r2 21305144 (after test2_40_r2);
the GPU jobs on the `a100|h100|h200|rtx6000` union with preemption opt-in and
requeue, no GPU type pinned. Records: `out/extra/test_draw2/` (the draw) and
`out/extra/evaluations/test2_*/`. Status (Slurm's full path, so no `$` reaches
the WSL shell):

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch "/opt/slurm/bin/sacct -j 21305137,21305138,21305139,21305140,21305142,21305143,21305144 -X -o JobID%12,JobName%10,State%12,Elapsed%12,NodeList%10"
```

### 8.2 The draw (fp-xdraw 21305137, 2026-10-29)

The draw ran on cs648 from 02:12 to 03:42 UTC (1 h 30 min) from 8e07f94, whose
`psse_env/` and `scripts/` are identical to the cell's recorded source 01ac598
(`git diff --stat 01ac598 8e07f94 -- psse_env scripts` is empty). Receipt
`out/extra/test_draw2/draw.done` (contract `conventional_only_test_draw_done_v1`;
development set sha256 `b409fbec2f45a45ac3838f56e46f3a49a539f11356da81d5572fbb404f3b9288`).

**References.** The collector read 61 files, none missing or unreadable, and
found 8,122 distinct roots: 13 expert aggregates (the D0 of 14 cells, two of
which, 20260908 and 20261013_ieee14_reverse, reuse another cell's file through
a symlink, and the control's), 14 development suites, 27 round suites, this cell's
validation and hard suites, the four renderings of the 2026-10-15 hard draw,
and the local preflight's development-only roots, which add 42 roots no other
file holds (the cancelled launch read 8,080 without them). 60 of the files are
readable by the disjointness check as reference suites; the preflight file holds
fingerprints only and enters the check through the collected-roots file.

**The builder.** It protected 8,239 roots: the 8,122 collected and the 117 of
the frozen BC0 suite and trace validation set that stage 0 protected (the
first draw's builder protected those 117 only, besides the cell's D0). With
multiplier 3 it stopped in the training allocation, which fills the two round
files it writes as a by-product (24 R/X roots between them): "Scenario family
'parameter' has 23 fresh roots; needs 24" (`build_m3.stderr`), that is, 49 of
the 72 R/X candidates built for that stream were roots of earlier sets. By the
rule fixed in 8.1 the draw was repeated once with multiplier 6, which built
1,152 development and 888 training candidates (the first draw: 576 and 444)
and completed. The development allocation shuffles each family's fresh
candidates with a fixed seed and keeps the first 32, 24 or 16
(`allocate_scenarios` in `scripts/run_dagger_research.py`; the development
stratum plan is empty), so the larger pool changes which candidates are kept,
not their kind.

**Sampling frame.** Both draws build candidates from the same part of the
tabular corpora: the 80% "train" partition of a sha256-modulo-5 split (494 R/X
rows, 418 meter rows, 2,801 clean rows; the evaluation fifth is used by
neither draw). The first draw excluded only this cell's D0 and stage 0's 117
protected roots; the second excludes every root any earlier set used, so it
samples the rows no earlier study touched. For R/X roots that remainder is
about a third of the admissible rows.

**Disjointness** (`disjointness_check.json`): passed. The development set (192
roots) and the two by-product round files (74 each) share no root with each
other or with the cell's D0 (370 roots), and the development set shares none
with any of the 60 reference files or with the collected roots (61 checks, all
zero).

**Composition and suite statistics** (`suite_stats.md`, both draws side by
side). The family counts are exactly the first draw's. Where the draws differ:

| | Draw 1 | Draw 2 |
| --- | --- | --- |
| R/X roots: dominant / ambiguous / misranked | 18 / 11 / 3 | 21 / 9 / 2 |
| R/X dominance ratio, median | 1.36 | 1.64 |
| R/X roots with R unchanged / X unchanged | 9 / 9 | 14 / 6 |
| Meter + R/X roots: dominant / ambiguous / misranked | 26 / 6 / 0 | 27 / 4 / 1 |
| Breaker roots: bus split / dangling terminal | 7 / 25 | 12 / 20 |
| Multi-meter roots on active-injection or voltage meters | 2 of 24 | 9 of 24 |
| Multi-meter roots with 2 / 3 / 4 / 5 bad meters | 9 / 4 / 5 / 6 | 8 / 3 / 6 / 7 |

Every multi-meter root puts all its bad meters on one kind of meter (all
active injections, all voltage magnitudes, all from-end reactive flows, and so
on), so the channel mix of that family is a count over 24 roots, not over the
80 or 84 meter errors. The rest is alike: meter errors of 10.1 to 16.0 sigma
(medians 12.2 to 12.9) on the single- and multi-meter roots and 11.0 to 29.8
sigma (medians 17.5 to 19.9) on the mixed roots; on the breaker families,
load scales 0.81 to 1.24 and every operating point an AC OPF; parameter scans
(8 per root) on every R/X and meter + R/X root and substation telemetry on
every breaker and meter + breaker root, as in the first draw. Exact two-sided
Fisher tests, draw 1 against draw 2: R/X dominant 18 against 21 of 32, p =
0.61; meter + R/X dominant 26 against 27, p = 1; bus splits 7 against 12 of
32, p = 0.27; multi-meter roots on active-injection or voltage-magnitude
meters 2 against 9 of 24, p = 0.036 (unadjusted, one of five comparisons; with
the seven meter kinds equally likely about 7 of 24 are expected, so the first
draw's 2 is the unusual count; counting reactive injections too, 6 against 13,
p = 0.075); single-meter roots on active-injection or voltage-magnitude meters
6 against 7 of 32, p = 1.

**Recurring error locations.** "Fresh" means an unseen physical realization,
not an unseen error location. The generator builds every meter root from a row
of a fixed measurement-error corpus and every R/X root from a row of a fixed
parameter corpus, and the corpora repeat locations at different operating
points; IEEE 14 has 20 lines and a short breaker list. Of the second draw's
roots, the error location (the bad-meter set, the line, the breaker, or both
for a mixed root) also occurs in the first draw for 13 of 32 single-meter, 3 of
24 multi-meter, 31 of 32 R/X, 17 of 32 breaker, 1 of 32 meter + R/X and 2 of 24
meter + breaker roots. Each is a distinct root (its own operating point, noise
and error values) and passes every disjointness check, as the first draw's
roots did against the training sets.

### 8.3 The evaluations (2026-10-29)

The six evaluations completed with exit 0, each on an RTX 6000 node and none
preempted (UTC, `sacct -D`):

| Job | Evaluation | Node | Ran | Elapsed |
| --- | --- | --- | --- | --- |
| 21305138 | `test2_40`: teacher, BC0, R1 at 40 actions | gr103 | 03:48 to 05:38 | 1 h 50 min |
| 21305139 | `test2_40_r2`: R2 at 40 actions | gr102 | 04:02 to 05:01 | 59 min |
| 21305140 | `test2_budget12` | gr103 | 05:39 to 07:14 | 1 h 34 min |
| 21305142 | `test2_budget12_r2` | gr102 | 05:02 to 05:53 | 51 min |
| 21305143 | `test2_budget20` | gr106 | 05:41 to 07:24 | 1 h 43 min |
| 21305144 | `test2_budget20_r2` | gr104 | 05:38 to 06:37 | 59 min |

A seventh job, added at 04:10 UTC after the launch, re-ran the cell's own
40-action evaluation of the first draw with `evaluate_suite.py` (fp-xeval
21315029, spec `test_repro40`; section 8.8). The adapters of every evaluation
match the stage receipts (`identities` in `pooled_tables.json`: the six new
evaluations and the four first-draw budget evaluations, all
`adapters_match_receipts: true`), and the rows evaluated are the receipt's
development set. `pooled_tables.py` ran on the login node at 07:25 UTC (from
`$PIPE/source` after `pipeline_environment`) and wrote
`out/extra/pooled_tables.{json,md}` and `pooled_tables_episodes.jsonl`
(4,608 records: two draws, three horizons, four policies, 192 roots).

### 8.4 Results on the second draw (192 roots)

At the cell's horizon (40 actions), every policy on the same roots and
measurement realizations, seed 20260912; successes, with safety-clean
successes in parentheses:

| Cohort | N | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- | --- |
| Single bad meter | 32 | 32 (32) | 32 (32) | 32 (32) | 32 (32) |
| Two to five bad meters | 24 | 23 (23) | 23 (18) | 23 (23) | 23 (21) |
| Line R/X error | 32 | 32 (32) | 32 (32) | 32 (32) | 31 (30) |
| Breaker status error | 32 | 32 (32) | 32 (32) | 32 (32) | 32 (32) |
| Meter + R/X error | 32 | 32 (32) | 32 (29) | 32 (32) | 32 (31) |
| Meter + breaker status | 24 | 24 (24) | 24 (19) | 24 (24) | 23 (22) |
| Clean controls | 16 | 16 (16) | 16 (16) | 16 (16) | 16 (16) |
| **All test roots** | 192 | **191 (191)** | **191 (178)** | **191 (191)** | **189 (184)** |
| Episodes with an invalid action (invalid actions) | | 0 (0) | 14 (22) | 0 (0) | 7 (19) |
| Episodes with a false commit (false commits) | | 0 (0) | 3 (7) | 0 (0) | 2 (5) |
| Loop episodes | | 0 | 1 | 0 | 1 |
| Mean actions per episode | | 9.83 | 10.13 | 9.81 | 9.95 |
| Mean WLS calls per episode | | 2.51 | 2.55 | 2.51 | 2.52 |
| Bounded comparisons | | 22 | 24 | 16 | 16 |
| Misranked parameter roots (3) | | 3 | 3 | 3 | 2 |
| Ambiguous parameter roots (13) | | 13 | 13 | 13 | 13 |

Per family, episodes with an invalid action / with a false commit / with a
loop, and mean actions and WLS calls:

| Cohort | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| Single bad meter | 0/0/0, 7.1, 2.00 | 0/0/0, 7.1, 2.00 | 0/0/0, 7.1, 2.00 | 0/0/0, 7.1, 2.00 |
| Two to five bad meters | 0/0/0, 20.3, 4.38 | 6/0/1, 20.4, 4.38 | 0/0/0, 20.4, 4.38 | 3/0/0, 20.5, 4.38 |
| Line R/X error | 0/0/0, 7.4, 2.00 | 0/0/0, 7.3, 2.00 | 0/0/0, 7.3, 2.00 | 1/1/0, 7.3, 2.00 |
| Breaker status error | 0/0/0, 8.0, 2.00 | 0/0/0, 8.1, 2.00 | 0/0/0, 8.0, 2.00 | 0/0/0, 8.0, 2.00 |
| Meter + R/X error | 0/0/0, 11.2, 3.00 | 3/3/0, 12.6, 3.22 | 0/0/0, 11.2, 3.00 | 1/1/0, 12.0, 3.12 |
| Meter + breaker status | 0/0/0, 12.0, 3.00 | 5/0/0, 12.4, 3.04 | 0/0/0, 12.0, 3.00 | 2/0/1, 11.9, 2.92 |
| Clean controls | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 |

**The root every policy fails** (r0_ae01c73aa1ff, two to five bad meters).
Five bad voltage-magnitude meters, at buses 1, 6, 12, 13 and 14, each 0.011 to
0.014 pu off (10 to 14 sigma). The teacher corrects bus 1 (J from 449 to 290)
and bus 14 (J to 177; threshold 130), and the verifier accepts both as partial
fixes. The other three errors, at buses 6, 12 and 13, are all low by 0.011 to
0.012 pu, and the estimate absorbs most of them as a lower voltage profile in
that area: the largest normalized residual falls to 4.33 against the
threshold 4.0. The teacher escalates after 15 actions
(`operator_escalation:recovery_options_exhausted`); its escalation audit lists
no supported recovery target and no further evidence. R1 takes the same 15
actions; BC0 and R2 escalate after 18 and 16 actions with one invalid action
each. The teacher's residual procedure cannot identify the remaining three
meters from this snapshot; the root stays in the test set and counts as a
failure for every policy. It is one of the nine multi-meter roots on
active-injection or voltage meters (8.2); the teacher solves the other eight,
among them another root with five bad voltage meters (in 33 actions).

**R2's other two failures.** A misranked R/X root (r0_d61da247b0a2): R2 does
not compare; it corrects the ranking's top line (line 3), whose refit brings J
from 683 to 103, under the threshold, so the verifier accepts the wrong line
and R2 hands off after 7 actions. The teacher and R1 compare lines 3 and 6 and
correct line 6 (8 actions). On the first draw every policy solved every
misranked root with one comparison. A meter + breaker root (r0_d0c20d4b4c0b):
R2 calls `correct_topology` with the breaker but without the line (the
teacher passes both), is refused, repeats the identical call, and the
evaluator stops the episode after 5 actions with a loop; its first-draw
breaker failure (r0_b7a5d222a6fe) also ended in the loop rule after a refused
`correct_topology`. R2 solves this root at 12 and 20 actions.

**Violations.** BC0: 14 episodes (22 invalid actions; 7 false commits in 3
episodes; 1 loop): five meter + breaker roots with one invalid action each,
three meter + R/X roots with invalid actions and false commits (2, 1 and 4
false commits), and six multi-meter roots (one with a loop, one the root every
policy fails); all solved but that one. R2: 7 episodes (19 invalid actions; 5
false commits in 2 episodes; 1 loop): three multi-meter roots (the shared
failure among them), two meter + breaker roots (its failure among them), a
meter + R/X root (11 invalid actions and 4 false commits, solved in 37
actions) and an R/X root (one invalid action and one false commit, solved).
R2 had no false commit on the first draw. The teacher and R1: none.

**Under the action budget** (second draw): successes (safety-clean), then
violation episodes (with an invalid action / a false commit / a loop) and the
episodes that ran to the budget, then the 80 multi-error roots:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 176 (176) | 173 (166) | 176 (176) | 173 (170) |
| 20 | 181 (181) | 180 (169) | 181 (181) | 179 (175) |
| 40 | 191 (191) | 191 (178) | 191 (191) | 189 (184) |

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 0; 51 | 10 (10/2/0); 60 | 0; 53 | 5 (5/3/0); 55 |
| 20 | 0; 2 | 14 (14/3/1); 13 | 1 (1/0/1); 10 | 6 (6/2/0); 12 |
| 40 | 0; 0 | 14 (14/3/1); 0 | 0; 0 | 7 (7/2/1); 0 |

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 64 (64); 0 | 61 (54); 10 | 64 (64); 0 | 62 (60); 4 |
| 20 | 69 (69); 0 | 68 (57); 14 | 69 (69); 1 | 68 (65); 5 |
| 40 | 79 (79); 0 | 79 (66); 14 | 79 (79); 0 | 78 (74); 6 |

The 112 single-error and clean roots are solved by every policy at every
budget except R2's misranked R/X root (111), and safety-clean except one more
R2 episode per budget (110). The roots not solved under a budget are
multi-meter roots for every policy (16 of 24 at 12 actions; at 20, 11 for the
teacher, R1 and R2 and 10 for BC0), for BC0 also meter + R/X roots (3 at 12
actions, 2 at 20) and for R2 meter + R/X roots (2 and 1), the misranked R/X
root and, at 40 actions only, the meter + breaker root above. Mean actions;
mean WLS calls:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 8.76; 2.33 | 8.85; 2.35 | 8.76; 2.34 | 8.77; 2.35 |
| 20 | 9.39; 2.43 | 9.68; 2.47 | 9.41; 2.43 | 9.49; 2.44 |
| 40 | 9.83; 2.51 | 10.13; 2.55 | 9.81; 2.51 | 9.95; 2.52 |

R1 equals the teacher on this draw at every horizon, root for root, in success
and in safety-clean success. Its one violation episode is at 20 actions on
the root every policy fails: after the teacher's first 14 actions it requests
repeated measurement scans (not available under `scada_only`: an invalid
action), asks for the topology and parameter contexts again (a loop), and
escalates after 18 actions. At 40 actions it escalates directly, as the
teacher does; the measurement context reports the remaining budget, so the
two horizons give it different observations. It is R1's first process
violation on any test root of this cell (the first draw: none at 12, 16, 20 or
40 actions).

### 8.5 Pooled over both draws (384 roots)

At 40 actions; successes (safety-clean):

| Cohort | N | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- | --- |
| Single bad meter | 64 | 64 (64) | 64 (64) | 64 (64) | 64 (64) |
| Two to five bad meters | 48 | 47 (47) | 46 (38) | 46 (46) | 46 (40) |
| Line R/X error | 64 | 64 (64) | 64 (64) | 64 (64) | 63 (62) |
| Breaker status error | 64 | 64 (64) | 64 (64) | 64 (64) | 63 (63) |
| Meter + R/X error | 64 | 64 (64) | 63 (54) | 64 (64) | 64 (62) |
| Meter + breaker status | 48 | 48 (48) | 47 (34) | 48 (48) | 47 (43) |
| Clean controls | 32 | 32 (32) | 32 (32) | 32 (32) | 32 (32) |
| **All test roots** | 384 | **383 (383)** | **380 (350)** | **382 (382)** | **379 (366)** |
| Episodes with an invalid action (invalid actions) | | 0 (0) | 33 (61) | 0 (0) | 16 (34) |
| Episodes with a false commit (false commits) | | 0 (0) | 6 (21) | 0 (0) | 2 (5) |
| Loop episodes | | 0 | 3 | 0 | 3 |
| Mean actions per episode | | 9.85 | 10.26 | 9.83 | 9.94 |
| Mean WLS calls per episode | | 2.50 | 2.57 | 2.50 | 2.51 |
| Bounded comparisons | | 47 | 46 | 36 | 37 |
| Misranked parameter roots (6) | | 6 | 6 | 6 | 5 |
| Ambiguous parameter roots (30) | | 30 | 30 | 30 | 30 |

Per family, episodes with an invalid action / a false commit / a loop, mean
actions and mean WLS calls:

| Cohort | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| Single bad meter | 0/0/0, 7.0, 2.00 | 0/0/0, 7.1, 2.00 | 0/0/0, 7.0, 2.00 | 0/0/0, 7.0, 2.00 |
| Two to five bad meters | 0/0/0, 20.5, 4.35 | 9/0/1, 20.4, 4.35 | 0/0/0, 20.5, 4.35 | 7/0/0, 20.7, 4.35 |
| Line R/X error | 0/0/0, 7.5, 2.00 | 0/0/0, 7.4, 2.00 | 0/0/0, 7.4, 2.00 | 1/1/0, 7.4, 2.00 |
| Breaker status error | 0/0/0, 8.0, 2.00 | 0/0/0, 8.2, 2.00 | 0/0/0, 8.0, 2.00 | 1/0/1, 8.0, 1.98 |
| Meter + R/X error | 0/0/0, 11.2, 3.00 | 10/5/0, 12.8, 3.30 | 0/0/0, 11.2, 3.00 | 2/1/0, 11.6, 3.08 |
| Meter + breaker status | 0/0/0, 12.0, 3.00 | 14/1/2, 13.0, 3.17 | 0/0/0, 12.0, 3.00 | 5/0/2, 12.1, 2.98 |
| Clean controls | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 | 0/0/0, 2.0, 1.00 |

Under the budget (the 12- and 20-action rows): successes (safety-clean); then
violation episodes (invalid / false commit / loop) and episodes that ran to
the budget; then the 160 multi-error roots, successes (safety-clean) and
violation episodes:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 353 (353) | 345 (327) | 352 (352) | 348 (341) |
| 20 | 362 (362) | 359 (334) | 361 (361) | 358 (350) |
| 40 | 383 (383) | 380 (350) | 382 (382) | 379 (366) |

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 0; 104 | 23 (23/3/1); 117 | 0; 106 | 10 (10/4/1); 110 |
| 20 | 0; 2 | 30 (30/6/2); 27 | 1 (1/0/1); 21 | 13 (13/2/2); 23 |
| 40 | 0; 0 | 33 (33/6/3); 2 | 0; 0 | 16 (16/2/3); 0 |

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 129 (129); 0 | 121 (104); 22 | 128 (128); 0 | 125 (120); 8 |
| 20 | 138 (138); 0 | 135 (110); 30 | 137 (137); 1 | 136 (129); 11 |
| 40 | 159 (159); 0 | 156 (126); 33 | 158 (158); 0 | 157 (145); 14 |

The 224 single-error and clean roots: the teacher and R1 224 (224) with no
violation at every horizon; BC0 224 (223) at 12 actions (one violation
episode) and 224 (224) at 20 and 40; R2 223 (221) at 12 and 222 (221) at 20
and 40, two violation episodes each. Mean actions; mean WLS calls:

| Budget | Expert | BC0 | R1 | R2 |
| --- | --- | --- | --- | --- |
| 12 | 8.77; 2.33 | 8.84; 2.35 | 8.76; 2.34 | 8.78; 2.35 |
| 20 | 9.37; 2.42 | 9.71; 2.48 | 9.40; 2.42 | 9.46; 2.43 |
| 40 | 9.85; 2.50 | 10.26; 2.57 | 9.83; 2.50 | 9.94; 2.51 |

### 8.6 Paired tests

Exact two-sided McNemar tests on the roots one policy fails and the other
solves (first policy only / second policy only; p). The second draw:

| Budget | BC0 vs R1, success | BC0 vs R1, safety-clean | BC0 vs expert, success | BC0 vs expert, safety-clean | R1 vs R2, success | R1 vs R2, safety-clean |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 3/0 (0.25) | 10/0 (0.0020) | 3/0 (0.25) | 10/0 (0.0020) | 0/3 (0.25) | 0/6 (0.031) |
| 20 | 2/1 (1) | 13/1 (0.0018) | 2/1 (1) | 13/1 (0.0018) | 0/2 (0.5) | 0/6 (0.031) |
| 40 | 0/0 (1) | 13/0 (2.4e-4) | 0/0 (1) | 13/0 (2.4e-4) | 0/2 (0.5) | 0/7 (0.016) |

Pooled (384 roots):

| Budget | BC0 vs R1, success | BC0 vs R1, safety-clean | BC0 vs expert, success | BC0 vs expert, safety-clean | R1 vs R2, success | R1 vs R2, safety-clean |
| --- | --- | --- | --- | --- | --- | --- |
| 12 | 7/0 (0.016) | 25/0 (6.0e-8) | 8/0 (0.0078) | 26/0 (3.0e-8) | 0/4 (0.125) | 0/11 (9.8e-4) |
| 20 | 5/3 (0.73) | 29/2 (4.6e-7) | 6/3 (0.51) | 30/2 (2.5e-7) | 0/3 (0.25) | 0/11 (9.8e-4) |
| 40 | 2/0 (0.5) | 32/0 (4.7e-10) | 3/0 (0.25) | 33/0 (2.3e-10) | 0/3 (0.25) | 0/16 (3.1e-5) |

The other pairs, pooled: R1 against the teacher 1/0 (p = 1) in success and in
safety-clean success at every horizon (the first draw's multi-meter root every
student misses); BC0 against R2 on safety-clean success 21/7 (p = 0.013),
24/8 (0.0070) and 26/10 (0.011) at 12, 20 and 40 actions, on success 6/3, 5/6
and 2/3; R2 against the teacher on safety-clean success 12/0 (4.9e-4), 12/0
(4.9e-4) and 17/0 (1.5e-5), on success 5/0 (0.0625), 4/0 and 4/0 (0.125). On
violations the pooled table has BC0 against R1 23/0, 29/0 and 33/0 episodes
and R1 against R2 0/10, 0/12 and 0/16 (`pooled_tables.md` lists every pair,
horizon and draw).

With Holm's correction over the 18 requested pooled tests (three pairs, two
endpoints, three horizons) all nine safety-clean tests stay significant (the
largest p is 9.8e-4) and no success test does (the smallest is 0.0078, the
tenth smallest p overall, against 0.05/9). On the second draw alone Holm keeps
the six BC0 safety-clean tests; R1 against R2 (p 0.016 to 0.031) survives it
only pooled. The 12-action success gap BC0 against R1 (7 to 0 pooled): four
meter + R/X roots where BC0's invalid actions and false commits spend the
actions R1 solves them in (11), two meter + breaker roots (one with an invalid
action and a loop, one clean that runs to the budget) and one multi-meter root
that runs to the budget; R1 solves each in 11 or 12 actions.

### 8.7 Draw 1 against draw 2

By the reading fixed in 8.1 no family is flagged on outcomes: per family and
policy the two draws' successes differ by at most one root. Unpaired exact
Fisher tests at 40 actions, draw 1 against draw 2: the teacher 192 against 191
(p = 1); BC0 189 against 191 (p = 0.62), safety-clean 172 against 178 (p =
0.37); R1 191 and 191, safety-clean 191 and 191; R2 190 against 189 (p = 1),
safety-clean 182 against 184 (p = 0.81). The largest per-family differences
in safety-clean success are BC0's: meter + R/X 25 against 29 of 32 (p =
0.30), meter + breaker 15 against 19 of 24 (p = 0.34). Under the budgets the
draws agree within two roots per policy (12 actions: teacher, BC0, R1, R2
177, 172, 176, 175 against 176, 173, 176, 173; 20 actions: 181, 179, 180, 179
against 181, 180, 181, 179). The differences in the suite statistics (8.2)
are flagged in 8.9.

### 8.8 Reproduction of the first draw's evaluation, and the GPU type

Why. The pooled table reads the first draw from the cell's round evaluations
(the DAgger runner) and the second from `evaluate_suite.py`. Job 21315029
re-ran BC0, R1 and the teacher on the first draw at 40 actions with
`evaluate_suite.py` and the same options, and
`out/extra/test_repro40_check.txt` compares it with the round-1 evaluation
root by root, through `followup_tables.records` (the record the tables read).

Result. The teacher and R1 are identical on all 192 roots in every field
(success, invalid action, false commit, loop, terminal outcome, actions, WLS
calls, comparisons). BC0 solves the same 189 roots; 6 roots differ in their
action count (by 1 to 6), and three of them in their violations: two lose
their one invalid action (r0_0225ec5e7698, meter + breaker; r0_117831cda301,
multi-meter) and one gains an invalid action and a false commit
(r0_0808023d323b, meter + R/X, 11 actions to 17, still solved). BC0's
safety-clean count is 173 instead of 172, its violation episodes 18 instead of
19 (false-commit episodes 4 instead of 3).

Where the differences come from. The job was preempted on an A100 node (ga011)
after 36 minutes and resumed from its per-episode checkpoints on an RTX 6000
node (gr105). BC0 ran 31 episodes on the A100 and 161 on the RTX 6000; all six
differing roots are among the 31, none among the 161. The cell's round-1
evaluation ran on an RTX 6000 (fp-r1e 20594224, gr103). R1 ran 32 episodes on
the A100 and 160 on the RTX 6000 and matches on all 192. Greedy decoding
reproduces on one GPU type; across GPU types BC0 takes a different action on
some roots (6 of 31 here) and R1 does not (R1 also scored its round-1 numbers
exactly in the round-2 evaluation, which finished on an H200).

For the tables. Every evaluation of the second draw ran on RTX 6000 nodes, as
did the first draw's 40-action evaluation of BC0, R1 and the teacher, its
12-action evaluation and R2's 12- and 20-action evaluations. R2's first-draw
40-action episodes come from two preempted RTX 6000 attempts and an H200
(fp-r2e 20594227), and the first draw's 20-action episodes of BC0, R1 and the
teacher from a preempted A100 attempt and an H200 (21070128). BC0's
safety-clean counts therefore carry about one root of evaluation variation;
no test above changes its conclusion by one root.

### 8.9 For the manuscript: what differs and what the pool changes

1. **The teacher fails one root.** The teacher column becomes 191 of 192 on
   the second draw and 383 of 384 pooled: the five-voltage-meter root of 8.4,
   which every policy fails. A statement that the teacher solves every test
   scenario no longer holds; the root is one the teacher's residual procedure
   cannot identify from one snapshot.
2. **R1 equals the teacher on the second draw** at every horizon, root for
   root, in success and safety-clean success (176, 181 and 191 at 12, 20 and
   40 actions); pooled, it trails the teacher by the first draw's multi-meter
   root at every horizon. R1 has one process violation in 1,152 pooled
   episodes (384 roots at three horizons): at 20 actions, on the root the
   teacher fails.
3. **BC0 matches the teacher in success on the second draw** (191), so pooled
   BC0 against R1 at 40 actions is 2 to 0 (p = 0.5) and the success gap stays
   not significant. At 12 actions it is 7 to 0 (p = 0.016 unadjusted; not
   significant under Holm over the 18 tests); five of the seven are BC0
   episodes with an invalid action or a false commit, the other two run to
   the budget, and R1 solves each in 11 or 12 actions.
4. **The safety-clean gap replicates**: BC0 against R1 13 to 0 on the second
   draw (p = 2.4e-4) and 32 to 0 pooled (p = 4.7e-10) at 40 actions, and
   significant at 12 and 20 actions.
5. **R2's regression replicates and widens**: R1 against R2 0 to 7 on the
   second draw (p = 0.016) and 0 to 16 pooled (p = 3.1e-5) at 40 actions. R2
   loses a misranked R/X root by skipping the comparison (pooled misranked
   roots: R2 5 of 6, every other policy 6 of 6) and makes its first false
   commits (2 episodes, 5 commits).
6. **Composition.** Multi-meter roots on active-injection or voltage meters 9
   against 2 of 24 (p = 0.036 unadjusted; the teacher's failure is one of
   them); bus splits 12 against 7 of 32 (p = 0.27; every policy solves every
   breaker-status root of the second draw); R/X strata 21/9/2 against 18/11/3
   (p = 0.61), meter + R/X 27/4/1 against 26/6/0.
7. **Sampling frame.** The second draw samples only corpus rows no earlier set
   used (for R/X roots about a third of the admissible rows) and needed
   candidate multiplier 6 (the rule fixed in 8.1). Error locations recur
   across the draws by construction (fixed corpora, 20 lines, a short breaker
   list); every root is a new physical realization, disjoint from every
   earlier set.
8. **Evaluation variation.** BC0's actions depend on the GPU type on a few
   roots (8.8); its pooled safety-clean count carries about one root of that
   variation. The second draw's evaluations ran on RTX 6000 GPUs only.

### 8.10 Records

On torch, under `out/extra/` of the cell: `test_draw2/` (the draw; the suite
in `suite/development.json`, sha256 `b409fbec...`), `evaluations/test2_40/`,
`test2_40_r2/`, `test2_budget12/`, `test2_budget12_r2/`, `test2_budget20/`,
`test2_budget20_r2/` and `test_repro40/` (each with `<slot>_eval.json`,
`summary.json`, `policies.json`, `comparison.json` and `episodes.jsonl`; the
console receipts beside them, the specs under `specs/`),
`pooled_tables.json` (sha256 `8bfcf67d...`), `pooled_tables.md`
(`cc83795d...`), `pooled_tables_episodes.jsonl` (`2dd6c91a...`) and
`test_repro40_check.txt`. Archive `/scratch/yx3882/test_draw2_records_20261029.tgz`
(2,429,657 bytes, sha256
`0ae2e56f928e611edd1df2e8873e200c02ad5cc27fb1bcbf511b277a3ec94ce4`): all of
these except the `<slot>_eval.json` files (about 70 MB each, kept on torch).
Copy of the archive and its extraction in the worktree, under
`output/conventional_only_20261022/results/test_draw2_20261029/` (gitignored,
as the section 7 copy). The per-episode records have the schema of section 7
(`episodes.jsonl` in every evaluation directory: physical root, scenario,
family, stratum, mimic, policy, success, invalid actions, false commits, loop,
terminal outcome, actions, WLS calls, comparisons);
`pooled_tables_episodes.jsonl` adds `draw` and `budget` to the same record,
one line per draw, horizon, policy and root.

## 9. Round 2 rerun under the 2026-10-29 round recipe

Why. The review in `docs/dagger_round2_regression_20261029.md` (sections 3
and 4) reads the recorded R2's regression (sections 5, 7 and 8: R1 against R2
0 to 16 on safety-clean success, pooled) as a consequence of the round
recipe: round 2 mixed its 432 new rows 1:1 with a uniform draw from the pool
of D0 and round 1's 437 rows, so most of round 1's rows were missing from R2's
pass, and R2 continued R1. The revised recipe trains round 2 on the
cumulative aggregate (`ROUND_AGGREGATE=cumulative`: every round-1 safe row
beside round 2's own, against a D0 sample of the same size) from the BC0
adapter (`ROUND_INIT=bc0`), and audits the mixture for exact label
conflicts. The paper session asked (2026-10-29, at the user's request) for
round 2 of this cell to be rerun under that recipe and for every R2 number
behind Table I to be refreshed. The expert, BC0 and R1 are not re-run: their
records stand (BC0's decoding varies with the GPU type, section 8.8).

### 9.1 Port and checks (2026-10-29)

**Merge.** origin/claude/eager-fermat-0ca9ms (43f9270: the recipe in
`scripts/run_dagger_research.py`, `pipeline.env`, `stage_collect.sbatch` and
`stage_train.sbatch`, its tests and the review) merged into this branch as
e3a7c00. The one conflict, in `overrides/conventional_only_20261022.env`,
keeps both appended blocks: the second test draw's settings, then the
review's comment-only recipe block. 91 tests pass
(`psse_env/dagger/test_research_dagger_minimal.py`,
`research/test_hpc_full_pipeline.py`).

**What the code does with round 1's rows.** `load_prior_onpolicy_rounds`
reads the exported chat rows of the named round-1 collection
(`d1.chat.jsonl`), requires each to be a row the round-1 ledger
(`d1.all.jsonl`) marks `research_label_eligible`, and re-renders each from
its ledger row under the current observation view, one row per exported row
(the quarantined ledger rows are never read). Their roots are excluded from
round 2's collection like D0's and checked against D0 and the protected
suites; the mixture keeps every one of them, adds round 2's safe rows, and
samples D0 to the same count (`D1_SHARE` 0.5); the exact label-conflict
audit runs over the whole mixture. The runner does not itself compare round
1's roots with the test, validation or hard roots, so the data were checked
on torch before training: `out/r1/collection/d1.chat.jsonl` holds 437 rows,
all marked eligible (of the ledger's 743), from 72 roots, every one of them
in the 74-root round-1 training suite and none in the round-2 training suite
(74 roots), the first test draw (192), the second (192), the validation
suite (96), the hard draw (193) or D0 (370). (The matched cloning control's
own aggregate, `out/extra/control_d0`, shares 4 roots with the round-1
training suite; it is not part of R2's mixture, and the control is not
re-run.)

**Additions** (0ddbc13, 5f7a467; 94 tests pass).
`submit_r2_rerun_20261029.sh` submits every R2 evaluation the cell recorded,
under its recorded name, rows and horizon, R2 alone, afterok on the new
training job, and refuses a name whose recorded directory, console receipt or
spec is still in place (`stage_eval_suite.sbatch` skips an evaluation whose
summary exists, so a record left in place would be read as the new R2's).
`followup_tables.py` takes `--r2-evaluation` and `--control-evaluation`, so
the round selection reads the new R2 from `{suite}_r2` and the control from
the set-aside `{suite}_r2_control` pair. `r2rerun_tables.py` compares the new
R2 with the recorded one and with the expert, BC0 and R1 on every set and
horizon (section 9.6 onward). `r2rerun_20261029/gate_mixture.sbatch` and
`finalize.sbatch` are the two cell-level jobs below, as deployed.

### 9.2 Deploy and configuration (22:31 UTC)

`deploy_cell_from_windows.sh` with BASE 8e07f94 (the cell source's HEAD),
OVERRIDES `conventional_only_20261022.env`, SUBMIT=0, commit 0ddbc13; the
dry-run prerequisites passed. The deployed `pipeline.env`, sourced as the
stages source it, resolves `ROUND_AGGREGATE=cumulative` and
`ROUND_INIT=bc0`; round 2 binds the collecting student to R1
(`out/r1/training/lora`), the training initialization to BC0
(`out/bc0/lora`), the pool to the D0 aggregate and the prior round to
`out/r1/collection`; the collection arguments name R1 as the adapter,
`out/r2/collection` as the output, `out/suite/r2_training.json` as the
training roots, the first test draw as the development roots,
`--d1-share 0.5` and `--prior-onpolicy-dir out/r1/collection`. What r2c, r2t
and r2e write: `out/r2`, the logs and `out/pipeline_summary.json`; the
round-2 pool build runs only under `pool`, and the training stage writes
`out/bc0/validation.subset.jsonl` only when it is missing (it is present).
A listing of `out/d0`, `out/bc0`, `out/r1` and `out/extra` (paths, sizes,
modification times) taken before the deploy
(`r2rerun_20261029/snapshot_predeploy_*.tsv` in the cell) is compared with
one taken after the last job (9.12).

### 9.3 Set aside and submitted (22:37 to 22:52 UTC)

Moved, never deleted (`mv -n`, refusing an existing target):
`out/r2` to `out/r2_pool_20261023`; `out/pipeline_summary.json` (r2e rewrites
it) to `out/pipeline_summary_pool_20261023.json`; the eight recorded R2
evaluations `test2_40_r2`, `test_budget12_r2`, `test_budget16_r2`,
`test_budget20_r2`, `test2_budget12_r2`, `test2_budget20_r2`,
`hard_r2_control` and `validation_r2_control`, each with its console receipt
and spec, to the same names with `_pool_20261023`; and the tables built from
them (`pooled_tables*`, `budget_tables*`, `followup_tables*` under
`out/extra`) likewise.

Jobs (`submitted_jobs.txt`): `FROM=r2c CHAIN='r2c r2t r2e'
ALLOW_PARALLEL_CHAIN=1 submit_pipeline.sh` gave r2c 21369734; the cluster
refused `scontrol hold` on the first r2t ("Unspecified error"), so that r2t
and its r2e (21369736, 21369739, both still pending) were cancelled and
resubmitted behind a gate: fp-r2gate 21369831 (CPU, afterok r2c;
`gate_mixture.sbatch` exits 0 only if the mixture's contract is the
cumulative one, `prior_onpolicy_selected` is 437,
`prior_onpolicy_retained_fraction` is 1.0 and the label-conflict audit finds
no conflicting observation or row), then r2t 21369832 (afterok gate) and r2e
21369834 (afterok r2t). The eight evaluations of the new R2 alone
(`submit_r2_rerun_20261029.sh`, afterok r2t, the recorded specs' rows,
horizons and the cell's seed and worker rule): `test2_40_r2` 21369835,
`test_budget12_r2` 21369837, `test_budget16_r2` 21369843, `test_budget20_r2`
21369846, `test2_budget12_r2` 21369847, `test2_budget20_r2` 21369848,
`validation_r2` 21369850 and `hard_r2` 21369851 (the first draw at 40
actions is r2e itself). fp-r2final 21370357 (CPU, afterok r2e and the eight;
`finalize.sbatch`) rebuilds the tables and archives the records.

### 9.4 How many round-1 rows the recorded R2 trained on

The review's count (section 4 of the review) on
`out/r2_pool_20261023/collection/round1.train.jsonl`: **57** of the 864 rows
of the recorded round-2 pass were round-1 rows (the review estimated 59).
Cross-check by example id against round 1's 437 exported rows: 57 distinct.
The recorded pass was 432 round-2 rows and 432 rows drawn uniformly from the
3,184-row pool of D0's training view (2,747) and round 1 (437), whose
expectation is 59.3; R2 trained 216 steps at 3e-5 from R1 and kept the final
checkpoint, on an NVIDIA RTX PRO 6000 Blackwell (job 20594226). So 380 of
round 1's 437 rows, 87%, were absent from R2's pass.
