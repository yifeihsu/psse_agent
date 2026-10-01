# Hypothesis ranking, step 5: the learned ordering in the ledger expert and the HPC cell (2026-10-01)

Step 5 of `docs/hypothesis_ranking_plan_20260930.md`. The step 4 decision was
to keep the physics rule as the admission gate and let the learned scores
order the ledger expert: when the rule admits a phasor acquisition on a root
the model considers unlikely to need an auxiliary stream, the ledger tries its
leading balanced hypothesis first, once, and acquires if that candidate is
rejected. This note records how that was built, the contract change it
needed, the paired CPU measurement of the plan's arms 1 to 3, and the cell
configuration for arm 4. Code on `codex/cleanup-20260928`; research only.

## 1. A contract conflict, stated and resolved (C5)

The ordering as recommended was not admissible under the controller as
written. The balanced screen mints `wls_hif_suspected` on every alarm its
HIF class wins; that signature carries the HIF marker, and the process oracle
refused every correction while any waveform-marked signature stood
(`correction_route_not_actionable`). The expert's combined stage applied the
same rule and kept only the diagnostics proposals. So on exactly the roots
where step 4 found the learned score useful, the same-sign flow-meter pairs
the screen mistakes for an HIF, a balanced correction could not be tried
before the phasors: the deferral would have been a silent no-op, and the
first version of the ranked expert returned no action at all on such a state.

The rule was written so that no fundamental-frequency "correction" could mask
a waveform event the screen had pointed at. Two things make the relaxation
defensible for the screen's own suspicion, and only for it:

- the suspicion is the physics rule's reason to request phasors, not a
  measurement of the network; sensor-reported waveform signatures and the
  phasor-confirmed HIF the NLM mints are measurements, and keep blocking;
- the deferral is bounded to one verified candidate per state, below a
  calibrated operating point, and verification judges the candidate. On the
  step 4 test split no HIF, unbalance or harmonic root with an HIF-won screen
  falls below that point (0 of 49 HIF roots; the two auxiliary roots below it
  are voltage-meter picks, which D3 holds anyway), so the exposure of a true
  waveform event to a wrong balanced commit is zero on that draw.

**C5 (2026-10-01).** `psse_env.actions.blocking_waveform_signatures` is the
waveform predicate for the correction gate: every waveform-marked signature
except the screen's `wls_hif_suspected`. The process oracle uses it in the
correction branch; the expert's combined stage uses it only while a deferral
is active (otherwise the suspicion keeps the stage on the diagnostic ladder,
as before, so the baseline and ledger experts' trajectories are unchanged).
Everything else that reads the HIF marker, the acquisition admission, the
NLM gate, the HIF continuation, the episode-level examined rule, is untouched.
The step 3 paired runs were repeated under C5 and reproduce the step 3
results root for root (section 3).

## 2. What was built

**The deployable ranker** (`psse_env/oracle/learned_ranker.py`). The study's
features used the full offline screen report; the runtime has the policy
observation. `policy_visible_features` reads what the policy sees and nothing
else: the WLS ledger of the active state (alarm flags, largest normalized
residual, residual breadth), the `wls_` residual and multiplier signatures
(which channel blocks stand out, the dominance tags, two flow meters of one
branch among them), the remaining anomaly score, and the compact screen report
(first and last compared round's class scores and winner, the accepted
sequence, each class's best refit objective and its ranked alternatives, the
channels set aside, the three suspicions). 73 features. Two provider details
made parity possible: the compaction of the screen report moved from the
provider into `psse_env.providers.hif_screen.compact_screen_report` so the
study and the runtime build the same object, and the never-filled
`dominant_residual_block` of the ledger (the breadth metric indexes slice
objects and always records none) is not read. A parity check on 13 freshly
built roots of eight families, the real environment's first WLS against the
study's pseudo-observation of the same root, finds all 73 features equal on
every root.

`research.hypothesis_ranking.ranker --feature-set policy` retrains the study
on these features and `--export-model` writes the fitted models as JSON for
`LearnedRanker`: the gradient-boosted trees themselves (node lists; the
export is checked against sklearn's probabilities on the test split before it
is written, agreement 1e-16), Platt scaling from the calibration split, the
operating-point thresholds, the test metrics and the provenance. No learning
library runs in the expert. The export is
`psse_env/oracle/models/learned_ranker_ieee14_20261001.json` (2.6 MB); a
model is refused on another network (bus count), which is what the IEEE 57
transfer result asked for.

On the policy-visible features the study's numbers hold (test split, 463
rows; `output/hypothesis_ranking_20260930/ranker_policy/report.md`):

| quantity | rule / screen | learned (policy features) |
| --- | --- | --- |
| `needs_aux` AUC | | 99.8% [99.6, 100] |
| recall on roots needing an auxiliary stream, at the rule's recall threshold | 98.5% [95.9, 100] | 99.2% (129 of 130) |
| acquisitions on roots needing none, same threshold | 12.0% [8.5, 15.7] | 2.1% [0.6, 3.9] |
| same-sign mimics flagged, at the screen's HIF recall | 37.5% [15.8, 61.1] | 0.0% |
| HIF recall there | 98.0% | 98.0% |
| unbalance against voltage-meter pick, AUC | | 97.3% [91.7, 100] |

The logistic alternative was also exported and rejected: at the rule's
recall threshold it flags 24.0% of the no-auxiliary roots and all six
HIF-won mimics, so it would defer nothing useful.

**The deferral** (`acquisition_deferral`, applied in
`ExpertPolicyOracle(hypothesis_ledger=True, learned_ranker=...)`). All of:
suspicion-gated profile; a valid current screen on the model's network; the
rule admits phasors now (HIF won, or a voltage channel set aside) and none
were requested on this state; no open candidate and no verified candidate
rejected on this state; the ledger's leading family is a balanced family with
an untested target that is not a voltage channel (D3); the model's `needs_aux`
probability is below the threshold that keeps the rule's recall on the
calibration split (0.0189 for this export). Then the suspicion stage is
skipped for this step, the combined stage runs under the ledger's ordering,
and the leading family's proposals carry
`learned_ranker_deferred_acquisition p_needs_aux=... threshold=...` so the
audit shows why. A rejected candidate ends the deferral and the rule's
acquisition follows unchanged; a failed execution ends it too.

**Variants and plumbing.** `psse_env/oracle/expert_variants.py` names the
teachers (`baseline`, `ledger`, `ledger_ranked`) and reads
`PSSE_EXPERT_VARIANT`; the aggregate generator, the DAgger collector, the
training-decision audit and the evaluation's expert arm all construct the
expert through it, and the research profile, the D0 receipt and the deploy
record carry the variant (a reused stage must declare the same one). The
harness `expert_e2e.py` gained `--expert ledger_ranked`, flow-meter pair
mimic roots (`--mimic-per-variant`, built as ordinary two-meter roots through
the generator's own admission and reported as their own families) and a
`deferred_acquisition` flag per episode; `compare_arms.py` reports it.

## 3. Paired evaluation (arms 1 to 3)

208 roots: the 160 development roots of step 3 (same cached draw, seed
20261001) plus 24 same-sign and 24 opposite-sign flow-meter pair mimics, a
40-step budget, the deployment verifier and the suspicion-gated contract. Arm
1 is the baseline expert, arm 2 the ledger expert, arm 3 the ledger expert
with the learned ranker. Outputs `output/hypothesis_ranking_20260930/step5_*`
and the paired reports `step5_compare_*.md` (gitignored).

| family | n | success 1 / 2 / 3 | mean steps 1 / 2 / 3 | phasors 1 / 2 / 3 | corrections 1 / 2 / 3 | rollbacks 1 / 2 / 3 |
| --- | --- | --- | --- | --- | --- | --- |
| no_error | 8 | 8 / 8 / 8 | 2.0 / 2.0 / 2.0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| telemetry_no_disturbance | 8 | 8 / 8 / 8 | 2.0 / 2.0 / 2.0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| measurement | 16 | 16 / 16 / 16 | 7.5 / 7.4 / 7.4 | 3 / 3 / 3 | 16 / 16 / 16 | 0 / 0 / 0 |
| multi_measurement | 12 | 12 / 12 / 12 | 23.5 / 19.0 / 19.2 | 3 / 3 / 3 | 45 / 45 / 45 | 0 / 0 / 0 |
| parameter | 16 | 15 / 15 / 15 | 7.2 / 7.2 / 7.2 | 0 / 0 / 0 | 17 / 17 / 17 | 0 / 0 / 0 |
| measurement+parameter | 16 | 16 / 16 / 16 | 11.0 / 11.0 / 11.0 | 0 / 0 / 0 | 32 / 32 / 32 | 0 / 0 / 0 |
| topology | 16 | 16 / 16 / 16 | 8.0 / 7.1 / 7.1 | 0 / 0 / 0 | 16 / 16 / 16 | 0 / 0 / 0 |
| measurement+topology | 12 | 12 / 12 / 12 | 12.0 / 11.0 / 11.0 | 0 / 0 / 0 | 24 / 24 / 24 | 0 / 0 / 0 |
| hif | 16 | 16 / 16 / 16 | 6.3 / 6.3 / 6.3 | 16 / 16 / 16 | 0 / 0 / 0 | 0 / 0 / 0 |
| measurement+hif | 8 | 8 / 8 / 8 | 13.0 / 11.9 / 11.9 | 8 / 8 / 8 | 13 / 10 / 10 | 5 / 2 / 2 |
| three_phase_unbalance | 16 | 16 / 16 / 16 | 4.0 / 4.0 / 4.0 | 16 / 16 / 16 | 0 / 0 / 0 | 0 / 0 / 0 |
| harmonic | 16 | 16 / 16 / 16 | 8.4 / 7.6 / 7.6 | 16 / 16 / 16 | 10 / 6 / 6 | 10 / 6 / 6 |
| mimic, same-sign pair | 24 | 20 / 20 / 20 | 14.2 / 12.2 / 11.1 | 10 / 10 / 1 | 48 / 48 / 48 | 0 / 0 / 0 |
| mimic, opposite-sign pair | 24 | 21 / 17 / 17 | 14.6 / 12.5 / 12.5 | 0 / 0 / 0 | 50 / 51 / 51 | 1 / 7 / 7 |
| **all** | 208 | 200 / 196 / 196 | 10.1 / 9.1 / 9.0 | 72 / 72 / 63 | 271 / 265 / 265 | 16 / 15 / 15 |

Spectra 18 / 18 / 18, false commits 0 / 0 / 0, deferred acquisitions
0 / 0 / 9.

**C5 changed nothing for the baseline and ledger experts.** On the 160
development roots arms 1 and 2 reproduce the step 3 arms root for root
(success, length, acquisitions, corrections and rollbacks all identical).

**The deferral does what it was built for, and nothing else.** Arm 3 differs
from arm 2 on ten roots. Nine are same-sign mimics whose screen flagged an
HIF: the ranker scored them at 0.004 to 0.006 against the 0.019 threshold,
the leading meter hypothesis was tried first, both meters were corrected and
committed, and the phasors were never requested (14 steps to 11, same
outcome on every one, including the two that fail the truth audit in every
arm). The tenth same-sign mimic with an HIF-won screen scored 0.022, just
above the threshold, and acquired as before. The tenth differing root is a
five-meter root where the deferral opened the three balanced contexts before
the phasors and no correction followed; the rule's acquisition came three
steps later with the same outcome. No root that needs an auxiliary stream
was deferred: HIF, mixed HIF, unbalance and harmonic roots have identical
trajectories in arms 2 and 3. In total the ranker saved 9 of the 72
acquisitions (all nine on roots that needed none) at no change in success,
corrections or rollbacks.

**The mimic roots expose a weakness of the ledger that the ranker does not
touch.** On the opposite-sign pairs (one flow meter biased up, the other
down, which on balanced SCADA resembles a branch whose two ends disagree) the
ledger expert solves 17 of 24 against the baseline's 21, with 7 rejected
candidates against 1. On 8 of the 24 the screen accepts a branch parameter
hypothesis first and only then two meters, and the meter targets it ranks
after that wrong branch are injection channels next to it, not the flow
meters; the ledger's first meter correction follows that ranking and misses
(first meter correction on the true target: 103 of 112 meter roots for the
ledger, 109 for the baseline, whose static residual order picks the flow
meters), and four of those eight roots are lost. The same-sign pairs are solved 20 of 24 by
every arm; the four that fail do so identically in all arms (both true meters
corrected and committed, the truth audit still refuses the final vector), so
they do not separate the arms. Neither mimic family is in the cell's plan;
the opposite-sign result is the first case where the ledger loses roots to
the baseline and is recorded as an open item for the screen (an opposite-sign
pair test, or the close-alternative comparison step 3 deferred).

## 4. The cell (arm 4)

`research/hpc/full_pipeline_20260907/overrides/hypothesis_ranking_20261001.env`
sets `EXPERT_VARIANT=ledger_ranked` with the suspicion-gated contract, the
2026-09-24 corpora, plans and seeds, and no reuse (`PREVIOUS_PIPE` empty: the
teacher changed, so D0, BC0 and both rounds are regenerated). `pipeline.env`
exports the variant to every stage; `prerequisites.sh` refuses the ranked
variant without the model export and loads it once; `deploy_remote.sh` and
the D0 receipt record it. A local stage 0 at the cell's settings with the
ranked teacher (16-root plan, 111 raw rows, every family exported,
`expert_variant` recorded in the provenance, no model-view leak) passed; the
remaining release findings are the known small-plan and local-build ones.

**Launch status.** The cell is prepared but not submitted. Submitting needs
the WSL SSH master to torch, which runs on the user's interactive NYU SSO
login; the master socket had expired and every batch SSH attempt in this
session was refused (`Permission denied (gssapi-keyex,...)`). The deployment
is scripted end to end (`research/hpc/full_pipeline_20260907/deploy_cell_from_windows.sh <commit>`: a local
clone of the 2026-09-24 cell's source as the new cell's source, the
incremental bundle from the last deployed commit uploaded through ssh stdin
and checksum-verified, `deploy_remote.sh` with the step-5 overrides and its
dry-run prerequisites, then `submit_pipeline.sh`), so once the master is back
(`scripts/start_torch_ssh_master.ps1`) one command launches the chain d0 ->
bc0 -> r1c -> r1t -> r1e -> r2c -> r2t -> r2e into
`/scratch/yx3882/research_full_pipeline_20261001_ranked`.

The evaluation summaries already report success by basis
(`summarize.py: success_basis`); under the suspicion-gated admission every
suite root alarms the WLS, so success conditional on an alarm is the
full-pipeline number for this cell. The student does not see the ranker's
probability: the deferral is a function of the observation it does see, and
whether the student learns it is part of what arm 4 measures.

## 5. Verification

`psse_env/oracle/test_learned_ranker.py` (features, model contract and
network guard, tree export against sklearn, deferral conditions, C5, the
ranked expert's first action and its acquisition after a rejection, the
variants and the tracked export); the ledger, screen, suspicion-gated
contract, WLS-gated boundary, executor-recovery, release-factory, evaluator,
research-DAgger and provider suites; `research/test_hpc_full_pipeline.py`
for the cell scripts. Final regression on the committed code: 579 passed (251 subtests) across
those suites, after an earlier 553-test pass before the provider change; the
cell's self-test 28 passed.

## 6. Reproduce

```bash
LOKY_MAX_CPU_COUNT=8 python -m research.hypothesis_ranking.ranker --dataset-dir output/hypothesis_ranking_20260930/ieee14_v3 --transfer-dir output/hypothesis_ranking_20260930/ieee57_v3 --output-dir output/hypothesis_ranking_20260930/ranker_policy --feature-set policy --export-model psse_env/oracle/models/learned_ranker_ieee14_20261001.json
```

```bash
PSSE_LOCAL_DIAGNOSTIC_BUILD=1 python -m research.hypothesis_ranking.expert_e2e --output-dir output/hypothesis_ranking_20260930/step5_ledger_ranked --expert ledger_ranked --seed 20261001 --roots-file output/hypothesis_ranking_20260930/step3_roots.json --mimic-per-variant 24 --plan '{"no_error": 8, "measurement": 16, "multi_measurement": 12, "parameter": 16, "topology": 16, "harmonic": 16, "hif": 16, "measurement+parameter": 16, "measurement+topology": 12, "measurement+hif": 8, "three_phase_unbalance": 16, "telemetry_no_disturbance": 8}'
```

The same command with `--expert baseline` and `--expert ledger` gives the
other arms; `compare_arms.py --a ... --b ... --output ...` pairs two of them.
