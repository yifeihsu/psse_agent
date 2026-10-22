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
[0.10, 0.50] or [2.04, 4.91]. `development.json` sha256 begins `fedf867685f2`;
the cluster draws the same file unless one of its 370 D0 roots collides with
a development candidate, which section 4 checks.

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

After stage 0, on torch: the cell's development suite against its own D0 and
rounds and every earlier cell's aggregate and suites (`check_suite_exclusion.py`
with `--reference-suite` for the 2026-10-01, 2026-10-07 and 2026-10-21 cells'
`out/d0/aggregate.raw.jsonl` and `out/suite/*.json`), and the development
file's sha256 against the local draw's (`fedf867685f2`).

Resume after a failure (receipts `out/d0.done`, `out/suite.done`,
`out/bc0.done`, `out/r{1,2}/collection.done`, `out/r{1,2}/training.done`,
`out/r{1,2}/round_summary.json`):

```bash
MSYS_NO_PATHCONV=1 wsl -- ssh torch "cd /scratch/yx3882/research_full_pipeline_20261022_conventional && FROM=<stage> ALLOW_PARALLEL_CHAIN=1 bash submit_pipeline.sh"
```

## 5. Results

PENDING (when the chain completes).
