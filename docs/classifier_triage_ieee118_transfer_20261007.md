# IEEE 14 triage GNN on IEEE 118: alarm-admitted data, zero-shot transfer, and where it fails (2026-10-07)

Question (2026-10-07): generate IEEE 118 HIF, unbalance and harmonic scenarios
the way the DAgger pipeline does, keep only the ones that trigger the WLS alarm,
drop the ones without a solvable operating point, and test whether the triage
GNN trained on IEEE 14 (`psse_env/oracle/models/triage_gnn_ieee14_20261004`,
the gate of `classifier_gated_diagnostics`) transfers. If recall is poor, find
where the problem is.

## Answer

The IEEE 14 GNN, weights and threshold unchanged, requests phasors on
**58.2% [55.7, 60.9]** of the IEEE 118 roots that need them (IEEE 14 test:
100%). It almost never requests them without a reason: 0.7% of rows that need
no phasors (IEEE 14: 0.9%). The AUC over all IEEE 118 rows is still 98.1%: the
ranking mostly transfers, but the IEEE 118 positives score far lower than on
IEEE 14.

| Family | IEEE 118 requested | Where the misses are |
| --- | --- | --- |
| HIF | 500/633 (79%) | strong faults: 52% requested below 1 pu of the local base, 97% at 2 to 4 pu (the IEEE 14 training range) |
| HIF with a bad meter | 422/633 (67%) | the same strong faults |
| Harmonic | 181/439 (41%) | no physical stratum separates hits from misses |
| Unbalance | 24/233 (10%) | everywhere (every VUF band, every phase-A share) |

On a missed root the family head almost always says "bad meter": measurement
for 130 of 133 missed HIFs, 199 of 209 missed unbalances and 190 of 258 missed
harmonics.

The problem lies in what the IEEE 14 training data covers, not in the WLS
evidence and not in the size-dependent inputs:

1. **The HIF strength range.** The pipeline's HIF recipe draws R in ohms
   (100 to 1,000 ohm). Every IEEE 14 training HIF sits on a 69 kV line, so
   training covers 2.1 to 6.3 pu of the local base. On IEEE 118 the same ohms
   on 138 kV and 345 kV lines give 0.09 to 5.2 pu: 61% of the admitted IEEE 118
   HIFs are stronger than any HIF the GNN saw. A strong single-phase fault
   spreads the balanced WLS residual over the region: a median of 105
   channels above 3 sigma on 70 assets, against 6 channels on 4 assets for an
   IEEE 14 HIF. The GNN reads that as gross meter errors. Scaling the
   residuals down into the IEEE 14 range recovers HIF recall to 97%
   (616/633), while unneeded requests stay at 0.9%. The HIF signature itself
   transfers; strong faults are simply outside the training distribution.
2. **Unbalance and harmonic cues are IEEE-14-specific.** On IEEE 118 a phase
   split gives a diffuse pattern: a median of 39 channels above 3 sigma on 29
   assets, with the top two holding 19% of the energy. An IEEE 14 unbalance
   gives 7.5 channels on 6 assets with the top two at 65%. No input change
   brings unbalance recall above 21% without flooding the negatives with
   requests. Harmonic roots have the same summary statistics on both
   networks, yet half are missed. Cropping each IEEE 118 alarm to an IEEE 14
   sized neighbourhood lifts harmonic recall to 79% and unbalance to 43%, but
   also lifts requests on the negatives roughly elevenfold (7.7% overall).
   The IEEE 14 model learned these two families from the compact patterns of
   one 14-bus network. On IEEE 14 its family heads tell them apart from a bad
   voltage meter only through that network's simulator artifacts (plan
   section 12). The IEEE 118 OpenDSS model has no such background: its healthy
   SCADA matches the AC-OPF to 4e-6 pu.
3. **Calibration.** The IEEE 14 threshold (0.278) is far too high for IEEE 118.
   Re-choosing it on the IEEE 118 calibration split for the same recall
   target gives 98.6% recall, but at 21.9% unneeded requests (IEEE 14: 0.9%).
   Recalibration alone cannot repair the transfer.
4. **Not the cause.** Re-expressing the chi-square ratio and the >3/>4 sigma
   fractions on IEEE 14's degrees of freedom and channel count (54.0%), or
   clamping degree, shunts and branch r, x, b, tap to the IEEE 14 range
   (58.2%), changes nothing.

**The architecture and the WLS evidence are not the bottleneck.** The same
GNN and training recipe, trained on the IEEE 118 train split, comes close to
the IEEE 14 operating point on the IEEE 118 test split. Trained on IEEE 14 and
IEEE 118 together, it holds both networks at once. Thresholds come from each
network's calibration split, at the IEEE 14 recall target of 99.3%.

| Trained on | Scored on | Recall | Unneeded requests | AUC |
| --- | --- | --- | --- | --- |
| IEEE 14 (the export) | IEEE 118 test, IEEE 14 threshold | 58.2% (all rows) | 0.7% | 98.1% |
| IEEE 118 | IEEE 118 test | 97.5% [95.3, 99.2] | 1.3% [0.0, 3.1] | 99.7% [99.2, 100.0] |
| IEEE 14 + IEEE 118 | IEEE 118 test | 97.2% [94.7, 99.2] | 0.7% [0.0, 1.7] | 99.9% [99.9, 100.0] |
| IEEE 14 + IEEE 118 | IEEE 14 test | 100.0% [100.0, 100.0] | 0.6% [0.0, 1.5] | 100.0% [99.9, 100.0] |

By family on the IEEE 118 test split, joint model: HIF 116/119, HIF with a
bad meter 114/119, unbalance 41/41, harmonic 79/81. Its negatives draw 2
requests out of 301 (one multi-meter root, one parameter root).

Caveat: the IEEE 118 splits are by window, so a line or bus seen in training
can reappear in the test split. The IEEE 14 location hold-out study lost 3
points of recall on unseen locations. A leave-one-network-out test (train on
IEEE 14 and IEEE 57, score IEEE 118) needs an IEEE 57 set with every family
and is not done.

## Reverse direction: trained on IEEE 118, scored on IEEE 14 (2026-10-08)

`python -m research.classifier_triage.transfer_ieee118_to_14` uses the same
GNN, features and recipe (five seeds). Training uses the IEEE 118 train
split only, early stopping uses IEEE 118 parents, and the request threshold
(0.272) comes from the IEEE 118 calibration split at the IEEE 14 recall
target. No IEEE 14 data enters. The model is scored on the IEEE 14 test
split: the 463 rows of the IEEE 14 benchmark.

| Model | Scored on | Recall | Unneeded requests | AUC |
| --- | --- | --- | --- | --- |
| trained on IEEE 118 | IEEE 118 test | 97.5% [95.3, 99.2] | 0.7% [0.0, 1.8] | 99.7% |
| **trained on IEEE 118** | **IEEE 14 test** | **92.3% [87.1, 96.6]** | **1.2% [0.3, 2.5]** | 98.2% [96.1, 99.6] |
| trained on IEEE 118, threshold from IEEE 14 calibration | IEEE 14 test | 98.5% | 48.9% | 98.2% |
| IEEE 14 export (trained on IEEE 14) | IEEE 14 test | 100% | 0.9% | 99.96% |
| trained on IEEE 14 + IEEE 118 | IEEE 14 test | 100% | 0.6% | 99.99% |
| *forward:* IEEE 14 export | IEEE 118 test | 56.7% | 0.3% | |

The transfer is asymmetric. Trained on IEEE 118, the GNN keeps most of the
IEEE 14 operating point; trained on IEEE 14, it loses 43% of recall on IEEE
118.

- **What transfers.**
  - Every harmonic root (50/50), every HIF (25/25) and 23 of 25 HIFs with a
    bad meter are requested.
  - No balanced root draws a request beyond 4 of 333: 2 bad meters, 1
    multi-meter root, 1 meter+parameter root.
  - That includes the 99 topology and meter+topology roots, a family the IEEE
    118 set does not contain: 0 requests.
  - The IEEE 118 threshold (0.272) lands next to the IEEE 14 export's (0.278).
- **What is lost: compact unbalances, 22 of 30 requested.**
  - Six of the eight misses touch 2 to 5 channels above 3 sigma, three of
    them voltage channels only (phase splits at buses 2 and 4, max |r| 7 to
    10). They look like one or two bad voltage meters, and the family head
    calls 5 of the 8 a meter.
  - The IEEE 118 training unbalances never look like that. Only 1.5% touch 5
    channels or fewer, and none are compact and voltage-led; 23% of the IEEE
    14 test unbalances are.
  - The misses score near zero. Re-choosing the threshold on IEEE 14 to
    recover them requests phasors on 49% of the negatives, so this is a
    ranking gap, not calibration.
- **False requests.** On the test split they are reactive-flow meters (Qt,
  13 to 14.5 sigma) and one voltage meter. On the IEEE 14 probes (a quiet
  window plus one bad meter), the IEEE 118 model requests phasors as follows,
  against the IEEE 14 model's rates on held-out parents:

  | Probe | IEEE 118 model | IEEE 14 model |
  | --- | --- | --- |
  | power meter, 10 to 20 sigma | 0.8-1.9% | 1.3-1.7% |
  | voltage meter, 10 to 20 sigma | 19-21% | 10-12% |
  | voltage meter, 4.5 to 9 sigma | 5-7% | 10-12% |

  There is no OpenDSS/OPF background difference.
- **Size-dependent inputs do not matter here either.** None of these changes
  moves recall or unneeded requests by more than 1.5 points:
  - globals re-expressed on IEEE 118's degrees of freedom and channel count:
    92.3% recall, 1.2% unneeded;
  - the IEEE 14 graph placed beside a quiet IEEE 118 window, so pooling
    dilutes as on IEEE 118: 90.8%, 1.2%;
  - the same union with globals recomputed over both: 90.8%, 1.2%;
  - branch r, x and tap clamped to IEEE 118's range: 92.3%, 1.2%.

Why the asymmetry: transfer works from the training set whose alarm patterns
cover the other network's.

- IEEE 118's admitted HIFs span 0.09 to 5.2 pu. 146 of its 378 training HIFs
  lie in IEEE 14's range (2.1 pu and above), so IEEE 14's compact HIFs are
  inside its distribution.
- IEEE 14's training HIFs never reach IEEE 118's strong, diffuse faults.
- The one IEEE 14 regime IEEE 118 lacks is the compact, voltage-only
  unbalance. On IEEE 14 the GNN separates it from a bad voltage meter only
  through simulator artifacts (plan section 12), so IEEE 118 physics alone
  may never supply it.

Outputs: `output/classifier_triage_20261008/ieee118_to_14`
(`transfer_ieee118_to_14.{json,md}`, `checkpoints/`).

## Error setup consistency (2026-10-11)

`docs/error_setup_audit_20261011.md` compares the two sets' error recipes,
their realized parameters and their WLS signatures family by family. In
short: the network-free parts match (meter sigmas, parameter factors, THD,
unbalance fractions, split and phase, load scale, noise); the HIF strength
(drawn in ohms), the chi-square branch of the admission rule and its feature
log(J / threshold), and the set composition (positive share, topology,
healthy-window sampling) do not; and at matched physical strength the
footprints still differ because of the networks themselves.

## Data (`output/classifier_triage_20261007/ieee118`)

`python -m research.classifier_triage.ieee118_dataset` writes rows in the
hypothesis-ranking study format. Every family uses its IEEE 14 recipe:

| Family | Recipe | Draws | Admitted | Alarm at margin 1.0 only | Quiet | Excluded |
| --- | --- | --- | --- | --- | --- | --- |
| HIF | 3-phase OpenDSS (138/161/345 kV bases, PV generators with Q limits) at the AC-OPF dispatch of loads U(0.80, 1.25); a resistive fault to ground at U(0.25, 0.75) along one of 175 same-voltage lines, phase uniform, R from 100-200 / 200-500 / 500-1000 ohm with equal weight | 700 | 633 | 53 | 3 | 11 |
| HIF with a bad meter | one power meter of an admitted HIF window offset by max(U(0.10, 0.30), 10 sigma x U(1, 1.5)), random sign | 633 | 633 | | | |
| Unbalance | same model and dispatch; one non-slack load bus split over its phases by Dirichlet(3, 3, 3), total kept | 500 | 233 | 78 | 187 | 2 |
| Harmonic | `build_trace` on the AC-OPF point; source a non-slack load bus, source THD U(0.10, 0.20), orders 5 to 19, true-RMS Vm | 500 | 439 | 42 | 19 | 0 |
| Measurement, multi-meter, parameter, measurement+parameter | the round-0 generator on 16 fresh IEEE 118 balanced corpora (AC-OPF windows, same load range), its own builders and admission | | 256 / 256 / 256 / 255 | | | |
| Healthy windows | OpenDSS references of the waveform windows (two noise draws each) and the corpora's clean windows (four draws each), kept when the WLS alarms | | 288 | | | |
| Flow-pair mimics | both flow meters of an eligible line biased 10 to 15 sigma, same or opposite sign | | 224 | | | |

- **Admission:** the pipeline's discovered-mode rule. Chi-square above 1.25
  times chi2(0.99, 863) = 962.6, or max |r| >= 5. Healthy windows and mimics
  are kept when they alarm at margin 1.0 (the population the gate runs on).
  SCADA sigma is 1e-3 on Vm and 1e-2 on power, as in the IEEE 14 corpora.
- **Exclusions:** `attempts.jsonl` records every draw and why it stopped.
  - 9 HIFs on 345 kV lines at 118 to 281 ohm have no operating point: the
    homotopy stalls, or a generator leaves its constant-power band.
  - 2 HIFs on 138 kV lines (134 and 324 ohm): OpenDSS would not accept the
    Newton state even at the fallback tolerance (below).
  - 2 unbalances (buses 90 and 112): the homotopy stalls at t = 0.86 and 0.94.
- **Phasors:** the noiseless three-phase voltages and branch currents of every
  admitted window are in `phasors.jsonl.gz`, for a later IEEE 118 run of the
  pipeline.
- **Size and time:** 3,473 alarmed rows, 24 minutes on 16 workers.

Two code fixes were needed:

- `Transmission/generate_hse_traces.build_trace` fixed the harmonic reference
  to row 0. It now uses the case's BUS_TYPE 3 row (IEEE 118: bus 69). IEEE 14
  traces are byte-identical.
- `three_phase_model/runtime._accept_state`. On OPF-dispatched IEEE 118
  circuits, OpenDSS's own iteration is unstable, which is why the Newton
  fallback runs. An unbalanced Newton state's KCL residual sits at the
  floating-point floor (about 1e-8 A on the stiff zero-sequence paths). OpenDSS
  sees that as a 5e-9 relative voltage change, above the model's 1e-12
  tolerance, iterates on and diverges.
  - A rejected state is now offered once more at 1e-7.
  - A state accepted that way has its generator P error measured relative to
    apparent power, at 1e-5. A synchronous condenser at P = 0 otherwise fails
    on 0.15 W.
  - States accepted at 1e-12 are unchanged. The fallback rescued 47 unbalance
    and 12 HIF windows; without it, 2 of 8 smoke unbalance windows had been
    excluded as "no operating point" although they solve.
  - The builder audits those 59 windows at 1e-5 pu (1 kW) instead of 1e-7 pu.

## Zero-shot results (`output/classifier_triage_20261007/transfer`)

| Set | Rows | Recall | Unneeded requests | AUC |
| --- | --- | --- | --- | --- |
| IEEE 14 test (same code path) | 463 | 100.0% [100.0, 100.0] | 0.9% [0.0, 2.1] | 100.0% [99.9, 100.0] |
| IEEE 118, all alarmed rows | 3,473 | 58.2% [55.7, 60.9] | 0.7% [0.3, 1.1] | 98.1% [97.8, 98.5] |

Intervals are 95% parent bootstraps.

Requests on the no-phasor rows:

| Family | Requested |
| --- | --- |
| Healthy windows | 1/288 |
| Measurement | 2/256 |
| Multi-meter | 6/256 |
| Parameter | 0/256 |
| Measurement+parameter | 0/255 |
| Mimics | 1/224 |

Alarm-only draws (margin 1.0, not admitted): HIF 49/53, unbalance 4/78,
harmonic 7/42.

HIF strata (requested / roots):

| Stratum | Requested |
| --- | --- |
| 138 kV | 498/611 |
| 345 kV | 2/22 |
| 100-200 ohm | 124/216 |
| 200-500 ohm | 217/244 |
| 500-1000 ohm | 159/173 |
| R below 1 pu | 111/213 |
| R 1-2 pu | 125/147 |
| R 2-4 pu | 228/235 |
| R 4 pu and above | 36/38 |
| max abs r below 10 | 220/227 |
| max abs r 10-20 | 155/175 |
| max abs r 20-40 | 117/191 |
| max abs r 40 and above | 8/40 |

The faulted phase does not matter. Strong faults are 31/71, 44/74 and 36/68
on phases A, B and C; faults of 2 pu and above are 103/106, 89/93 and 72/74.

Unbalance by VUF at the bus: below 1%: 0/23, 1-2%: 12/86, 2-4%: 12/86, 4% and
above: 0/38. Harmonic by maximum bus THD: below 0.2: 106/248, 0.2-0.3: 52/127,
0.3-0.5: 23/62.

Alarm signatures, as medians: channels above 3 sigma, assets they touch,
share of the top two in their squared residual, max |r|.

| Rows | Channels > 3 sigma | Assets | Top-2 share | Max abs r |
| --- | --- | --- | --- | --- |
| IEEE 14 HIF (train) | 6 | 4 | 0.58 | 7.1 |
| IEEE 118 HIF, requested | 19 | 15 | 0.41 | 11.0 |
| IEEE 118 HIF, missed | 105 | 70 | 0.27 | 29.3 |
| IEEE 14 unbalance (train) | 7.5 | 6 | 0.65 | 8.0 |
| IEEE 118 unbalance, requested | 23.5 | 19.5 | 0.22 | 7.6 |
| IEEE 118 unbalance, missed | 39 | 29 | 0.19 | 9.8 |
| IEEE 14 harmonic (train) | 15 | 10 | 0.44 | 11.5 |
| IEEE 118 harmonic, requested | 14 | 12 | 0.48 | 10.3 |
| IEEE 118 harmonic, missed | 17 | 14 | 0.39 | 10.4 |
| IEEE 118 bad meter | 5 | 5 | 0.80 | 10.6 |
| IEEE 118 healthy alarm | 4 | 4 | 0.60 | 4.2 |

## Where it fails: one change at a time

`python -m research.classifier_triage.transfer_ieee118` re-scores every row
with one change at inference.

| Change | Recall at the IEEE 14 threshold | Unneeded | AUC | Test recall, threshold re-chosen on IEEE 118 | Test unneeded, re-chosen |
| --- | --- | --- | --- | --- | --- |
| none | 58.2% [55.7, 60.9] | 0.7% | 98.1% | 98.6% | 21.9% |
| globals on IEEE 14's dof and channel count | 54.0% [51.5, 56.6] | 0.7% | 98.0% | 98.9% | 24.6% |
| 14-bus neighbourhood of the worst bus (with those globals) | 67.6% [65.6, 69.5] | 7.7% | 86.8% | 99.2% | 87.4% |
| residuals scaled to max 10.2 (IEEE 14 HIF q95) | 72.7% [70.6, 74.8] | 0.9% | 97.7% | 99.4% | 47.8% |
| static inputs clamped to the IEEE 14 range | 58.2% [55.8, 60.9] | 0.7% | 98.2% | 99.2% | 22.3% |
| all of them | 69.2% [67.3, 71.2] | 6.9% | 88.5% | 98.9% | 79.4% |

By family:

- **Residual scaling:** HIF 616/633, HIF with a bad meter 527/633, harmonic
  217/439, unbalance 48/233. Negatives are unchanged: 1 healthy window,
  3 bad meters, 9 multi-meter roots, 1 mimic.
- **The 14-bus neighbourhood:** HIF 568/633, harmonic 345/439, unbalance
  100/233, HIF with a bad meter only 298/633. Requests on the negatives rise
  to 31 healthy windows, 27 bad meters, 35 multi-meter roots and 17 branch
  roots.

## What would fix it

- **Train on the networks and strengths the gate will meet.** One joint IEEE
  14 + IEEE 118 model serves both networks at the IEEE 14 operating point, so
  the gate does not need one model per network. For a network without
  training data, broaden what the model sees:
  - draw HIF strength on the local per-unit base, or extend the ohm range down
    to the strong end, so strong faults are in training;
  - add networks whose unbalance and harmonic patterns spread over tens of
    assets, as on IEEE 118.
- **Calibrate the threshold on the network it gates.** IEEE 118 positives
  score far lower under the IEEE 14 model. Even the ceilings use a threshold
  from their own network's calibration split; the joint model's is 0.64 on
  IEEE 118 and 0.20 on IEEE 14.
- **Long-range patterns.** Unbalance and strong faults spread over 30 to 70
  assets on IEEE 118, past the three-hop receptive field. The ceilings show
  three hops are enough once such patterns are in training. On 14 buses three
  hops reach almost everything, so the IEEE 14 model never had to learn them
  locally.

## Agentic transfer: the IEEE 14 student on IEEE 118 (2026-10-11)

The question moved from the gate to the whole agent: does the Gemma 4 student
trained on IEEE 14 (the 2026-10-07 classifier-gated cell's round-2 adapter)
transfer zero-shot to IEEE 118, with the joint IEEE 14 + 118 triage GNN as
its gate, on every family but topology?  The cell is prepared
(`research/hpc/full_pipeline_20260907/overrides/ieee118_transfer_20261011.env`,
chain `d0 -> r1c -> zs`): stage 0 builds the IEEE 118 balanced corpus and the
expert aggregate, the development suite draws 124 roots of the nine
non-topology families, and the zero-shot stage evaluates the frozen student
beside the expert.  This section records what had to change for IEEE 118 to
run the waveform families at all, the local checks of the expert, and a rule
of the environment that does not transfer.

### What IEEE 118 needed

- **Waveform families on a registry system.**  The generator accepted HIF and
  unbalance corpora for IEEE 14 only; the registry now lists them for IEEE 118
  (`psse_env/systems/registry.py`), the harmonic synthesis runs on the
  registry case with its reference bus as the harmonic ground, and the
  balanced unbalance screen takes the system's line table.
- **PMU rungs on any case.**  The classifier-gated profile's HIF tools
  (zero-sequence line screen, PMU HIF estimate) read a line-parameter table
  that was hard-coded for IEEE 14; `line_parameters_from_case` builds it from
  the case (186 rows, 177 lines, 138 and 345 kV bases) and the provider passes
  it for any case but IEEE 14, whose legacy table is kept.
- **Corpora in the pipeline's row schema.**  `ieee118_dataset.py` writes
  `pipeline_corpus/{hif_samples,unbalance_samples}.jsonl`: the admitted
  windows of the IEEE 118 three-phase model at the AC-OPF dispatch, noisy and
  clean phasors at the PMU sigma 1e-4, one scan per HIF window.
- **Per-network gate thresholds.**  The runtime manifest carries
  `request_thresholds` by case; the joint model
  (`psse_env/oracle/models/triage_gnn_joint_20261011`, residual_v2 view,
  thresholds 0.096 on IEEE 14 and 0.338 on IEEE 118) is selected with
  `--triage-classifier` / `TRIAGE_CLASSIFIER` and reaches the provider through
  `PSSE_TRIAGE_CLASSIFIER`.
- **Pipeline plumbing.**  `pipeline.env` and `stage_d0.sbatch` take the
  waveform corpora of any system from `HIF_CORPUS_*` / `IMBALANCE_CORPUS`, the
  balanced corpus's load-scale range (`CORPUS_LOAD_SCALE_MIN/MAX`, set to the
  waveform corpora's U(0.80, 1.25) so the operating point does not tell the
  families apart) and the progress basis below.

### Local checks of the expert (14-root smoke suite, every non-topology family)

| gate | success | first-triage recall (6 HIF/unbalance roots) | unneeded requests (6 balanced roots) | wall |
| --- | --- | --- | --- | --- |
| tracked IEEE 14 model | 11/14 | 2/6 | 0/6 | 6.3 min |
| joint IEEE 14 + 118 model | 13/14 | 6/6 | 0/6 | 2.7 min |

The IEEE 14 gate refused phasors on both HIF roots at the first WLS (scores
below 0.001 against 0.278) and the expert spent its budget on meter edits; the
joint gate admits them and the HIF, HIF-plus-meter, unbalance and harmonic
roots all solve.  The remaining miss is a five-meter root, which exposes the
rule below.

### A rule that does not transfer: the partial-progress floor

After a correction the environment accepts a candidate as partial progress
only if its `global_progress`, the fraction of the parent's chi-square
statistic the correction removed, clears a floor (20%, or 10% for a singleton
repair inside a same-channel residual cluster).  The statistic is the anomaly
energy plus a noise baseline equal to the degrees of freedom: 95 on IEEE 14
(122 measurements, 27 states) and 863 on IEEE 118 (1098 and 235).  The error
setups are consistent (the audit above; the mean excess of a multi-meter root
is 298 on IEEE 14 and 283 on IEEE 118), so one correct repair removes about
the same energy on both systems but a much smaller fraction of the statistic
on IEEE 118.  On the five-meter root the first repair passed at 10.3% and the
next four correct repairs were rolled back at 9.5%, 7.7%, 7.6% and 7.3%, and
the expert escalated.

Measured on the multi-meter rows of a balanced corpus of each system
(`multi_gross_outliers`, two to five bad meters, repaired one at a time in
residual order; scratch script of this session):

| system | rows | dof | mean excess | repairs under the 10% floor, statistic basis | roots with one | same, excess basis | roots with one |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 14 | 80 | 95 | 298 | 3/288 | 3/80 | 0/288 | 0/80 |
| IEEE 118 | 120 | 863 | 283 | 313/432 | 115/120 | 4/432 | 4/120 |

Under the historical basis the family is nearly unsolvable on IEEE 118 for
the expert and therefore for any student; on IEEE 14 the same rule costs
almost nothing, which is why it was never noticed.  The environment now has
an opt-in `progress_basis` (`psse_env/transactional_env.py`,
`--progress-basis` on the research runner, the aggregate and the e2e script,
`PROGRESS_BASIS` in `pipeline.env`): `chi_square_statistic` is the unchanged
default; `chi_square_excess` measures the fraction of the excess over the
degrees of freedom of the parent solve, the network-size-independent form of
the same rule, and falls back to the statistic when the parent has no excess.
Replayed under it, the five-meter root is solved in 21 steps (five repairs,
each committed).  The IEEE 118 cell runs with `PROGRESS_BASIS=chi_square_excess`;
the IEEE 14 cells and their results are unaffected.

### The expert on the 124-root IEEE 118 development suite

The local suite (`research/classifier_triage/ieee118_suite.py`: a 540-row
balanced corpus at U(0.80, 1.25) and the regenerated waveform corpora, the
cell's development plan, 124 roots) rolled out with the joint gate under both
progress bases, then again with the localizer fix below:

| family | n | statistic basis | excess basis | excess basis + noise-floor fix |
| --- | --- | --- | --- | --- |
| no_error | 8 | 8 | 8 | 8 |
| measurement | 16 | 16 | 16 | 16 |
| multi_measurement | 12 | 6 | 12 | 12 |
| parameter | 16 | 16 | 16 | 16 |
| measurement+parameter | 16 | 16 | 16 | 16 |
| harmonic | 16 | 12 | 12 | 16 |
| hif | 16 | 16 | 16 | 16 |
| measurement+hif | 8 | 7 | 6 | 6 |
| three_phase_unbalance | 16 | 16 | 16 | 16 |
| **total** | 124 | 113 | 118 | 122 |

False commits 0 in every run; healthy meters touched 2, 3 and 3; the gate's
first-report recall 97.5% (39 of the 40 roots that need phasors) with no
unneeded request on the 68 balanced roots; 19-20 min per 124-root run.  The
multi-meter column is the progress-floor effect of the previous section (the
six statistic-basis losses are the five-meter and four-meter roots).  The IEEE
14 suite of the 2026-10-04 study rolled out under the same code gives the
same waveform-family results as before the changes (56/56 on harmonic, HIF,
HIF-plus-meter and unbalance, identical step counts), so none of this moves
the IEEE 14 cells.

**Harmonic roots and the unbalance localizer's noise floor.**  The four
harmonic misses had the expert finalize after the three-phase estimator: the
estimator reported a *significant* unbalance source at bus 10 with a
per-phase power spread of 0.07% and the expert, holding an explanation of the
alarm, finalized instead of asking for spectra.  Bus 10 carries 4 pu of
generation on one line.  Phasor noise is absolute, so the per-phase power
noise is `|V| sigma sqrt(n)` from the summed currents plus `|I| sigma` from
the bus voltage; the floor in `bus_shunt_power_unbalance` had only the first
term, which is fine on IEEE 14 (injections of about 1 pu) and four times too
low at bus 10.  The floor now carries both terms
(`three_phase_nlm/branch_current_analysis.py`, test
`tests/test_unbalance_noise_floor.py`); a real 5% unbalance clears it by a
factor of 20 and the sixteen IEEE 118 harmonic roots solve.

**HIF with a bad meter (6 of 8).**  One root is a gate miss: a 970 ohm fault
(7 MW on a 138 kV line) behind a bad injection meter scores 0.0075 at the
first WLS; the expert repairs the meter and then the fault line's own flow
meter, and the healthy edit fails the audit.  The other has the fault
estimated exactly (line 3-5, phase C, 140.7 ohm against 140.6) but the
balanced WLS conditioned on it still flags the line's terminal flow and a
nearby voltage as outliers, so the expert bundles them with the bad meter.
The conditioning residual grows with the fault power: 100-200 ohm at 138 kV
is 56 MW, four times the same resistance at IEEE 14's 69 kV, and the
balanced-model conditioning tolerance was set on IEEE 14 powers.  This is
left as a finding: the cell's student will see the same two misses in its
expert reference.

### The cell on torch

Committed as `5570bfc` and deployed to
`/scratch/yx3882/research_full_pipeline_20261011_ieee118` on 2026-10-11 with
the corpora uploaded beside it (sha256-verified).  The first chain (jobs
19579813, 19579814, 19579815) built the 2100-row IEEE 118 balanced corpus and
then aborted stage 0 in the expert aggregate: on one root the expert asked
for an operator escalation at step 5 and the training-label audit
(`assert_training_decision_evidence`) refused it with
`required_recovery_contexts_missing`.  The audit requires every context tool
implied by the unexplained signature families to have been consulted on the
same state; on IEEE 118 the balanced WLS leaves spurious
`wls_branch_multiplier line_status_or_parameter` signatures on healthy lines,
so after an HIF explanation and a meter edit the parameter and topology
contexts become "required" while the expert hands over with
`recovery_options_exhausted` after the measurement context alone.  A local
19-root aggregate run reproduced it once (an HIF root's counterfactual
branch); the three escalating HIF roots of the development suite verify.

The collector already knew how to end such an episode and keep the audit's
missing list; `03af043` exposes it (`--stop-episode-on-unverified-expert-label`
on the aggregate and the research runner, `UNVERIFIED_EXPERT_LABEL_POLICY`
in `pipeline.env`, `abort` by default) and writes the disagreements
(`d0/unverified_expert_labels.jsonl`, the collection's
`unverified_expert_labels/<root>.json`).  The cell runs with `stop`.  The
second chain (jobs 19583277, 19583278, 19583279) completed the aggregate
(92 roots, 686 rows, none quarantined, two unverified labels: an HIF root
and an HIF-plus-meter root, both `recovery_options_exhausted` with the
parameter and topology contexts required by leftover branch-multiplier
signatures) and then failed at the suite draw, because stage 0 hands the
suite builder the same system arguments as the aggregate and the builder
did not know `--progress-basis`; `bf22a8d` records it in the suite
manifest.  The third chain (jobs 19585506, 19585507, 19585508) was
submitted at 17:32 UTC and resumes stage 0 at the suite draw.
The teacher-side fix, consulting the contexts the audit requires before a
recovery handoff or dropping chance multiplier signatures from the
requirement, is open.

### Zero-shot result: the IEEE 14 student on the IEEE 118 development roots

The third chain completed at 21:07 UTC (stage 0 1 h 48 min for the suite
draw, the collection 16 min, the zero-shot evaluation 30 min on one H200).
The frozen round-2 adapter of the 2026-10-07 classifier cell, trained on
IEEE 14 only, rolled out with the joint gate on the 124 IEEE 118 development
roots beside the expert (`output/classifier_triage_20261011/cell_results/`
holds the summaries):

| family | n | student | expert | student false commits | student invalid actions | loops | spectra requested |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_error | 8 | 8 | 8 | 0 | 0 | 0 | 0 |
| measurement | 16 | 16 | 16 | 0 | 0 | 0 | 0 |
| multi_measurement | 12 | 11 | 12 | 1 | 3 | 1 | 0 |
| parameter | 16 | 15 | 15 | 0 | 0 | 0 | 0 |
| measurement+parameter | 16 | 15 | 16 | 2 | 4 | 1 | 0 |
| hif | 16 | 14 | 14 | 11 | 28 | 3 | 0 |
| measurement+hif | 8 | 6 | 6 | 0 | 15 | 3 | 0 |
| three_phase_unbalance | 16 | 16 | 16 | 0 | 10 | 3 | 0 |
| harmonic | 16 | 0 | 16 | 12 | 38 | 13 | 0 (expert 16) |
| **total** | 124 | **101** | **119** | 26 | 98 | 24 | 0 |

Every family but one transfers at the expert's level: the balanced families
within one root of the expert, HIF and HIF-plus-meter equal to it, unbalance
perfect.  The whole gap is the harmonic family, where the student solves no
root.  Its trace is the same on all sixteen: WLS alarm, phasors (admitted by
the gate), the three-phase estimator finds no unbalance and no HIF, and then
the student reads the measurement context and edits meters instead of asking
for spectra; the expert asks for spectra on all sixteen and localizes every
source.  The student spends 25 meter edits and 12 false commits on these
roots, touches a healthy meter in every one, and 13 of 16 end in the loop
detector.  On IEEE 14 the same adapter requests spectra on every harmonic
root, so the cue it learned for "balanced phasors plus a standing alarm means
spectra" does not survive the move: on IEEE 118 the alarm after the estimator
looks to it like the meter-error pattern it was trained to repair.  The
expert's rule is explicit (both acquisition tiers before any handoff) and
does not depend on the network.  The observation the student decides on
shows the cue that moved: on the IEEE 14 harmonic roots of the classifier
cell (round-1 student, 16 of 16 spectra requests) the WLS signatures after
the estimator are non-dominant outliers and multipliers spread over the
network (anomaly breadth median 0.15 of the channels, no `_dominant`
marker on most roots); on IEEE 118 the same distortion leaves five
dominant outliers in a breadth of 0.02, which is the footprint of a meter
error on IEEE 14.  The policy learned a network-relative cue for the
spectra request; a network-free cue (or the expert's explicit rule) is
what a transferable student needs.  The round-1 collection already holds
28 expert-labelled harmonic rows from IEEE 118, so a DAgger round on them
is the direct way to measure how much of the gap one round closes.

*Follow-up (2026-10-13).*  The triage GNN's family head already names the
waveform family of an admitted request (51/51 harmonic roots on IEEE 118
under the joint export); the runtime report now exposes it and a harmonic
call opens the spectra before the phasors, so the escalation the student
never learned is no longer a second step.  See
`docs/triage_family_scores_20261013.md`.

*Result (2026-10-13, cell A of the student cells).*  On the same 124
roots the frozen IEEE 14 student scores 115/124 under the spectra-first
environment with nothing retrained (harmonic 16/16 in four steps), one
DAgger round on IEEE 118 gives 114/124, and the expert stays at 119/124.
The adapted adapter scores 156/160 zero-shot on the IEEE 14 development
roots (expert 159/160), losing only the three unbalance roots the joint
gate sends to spectra first; the tables are in
`docs/triage_family_scores_20261013.md`.

The student also pays for its IEEE 14 habits on the HIF roots it still
solves: 11 false commits and 28 invalid actions across the sixteen (the
expert has none and six), meter edits committed on states whose alarm the
fault explains.  The collection stage, which ran the student on 24 training
roots with the expert labelling, measured a 24% tool disagreement on the 140
comparable states and a 1.4% invalid-action rate.

The expert's own 119 of 124 match the local run: the two HIF-plus-meter
misses of the local suite, two HIF roots (one invalid-action quarantine, one
escalation) and one parameter root.  Two aggregate episodes and one
collection episode ended at an unverifiable escalation label and were
recorded (`d0/unverified_expert_labels.jsonl`, `r1/.../unverified_expert_labels/`).

## Reproduce

```bash
python -m research.classifier_triage.ieee118_dataset --output-dir output/classifier_triage_20261007/ieee118 --workers 16
python -m research.classifier_triage.transfer_ieee118 --dataset-dir output/classifier_triage_20261007/ieee118 --output-dir output/classifier_triage_20261007/transfer --seeds 5
python -m research.classifier_triage.transfer_ieee118_to_14 --dataset-dir output/classifier_triage_20261007/ieee118 --output-dir output/classifier_triage_20261008/ieee118_to_14
python -m research.classifier_triage.transfer_ieee118_to_14 --dataset-dir output/classifier_triage_20261007/ieee118 --output-dir output/classifier_triage_20261008/ieee118_to_14 --probes-only
# 2026-10-11: the excess-energy admission rule and the residual_v2 view (docs/error_setup_audit_20261011.md)
python -m research.classifier_triage.ieee118_dataset --output-dir output/classifier_triage_20261011/ieee118_v2 --workers 16
python -m research.classifier_triage.transfer_ieee118 --dataset-dir output/classifier_triage_20261011/ieee118_v2 --output-dir output/classifier_triage_20261011/transfer_v2 --view residual_v2 --admission-rule excess_energy
# 2026-10-11: the agentic transfer (joint gate, pipeline corpora, local expert check, the HPC cell)
python -m research.classifier_triage.export_joint --dataset-dir output/classifier_triage_20261011/ieee118_v2 --output psse_env/oracle/models/triage_gnn_joint_20261011
python -m research.classifier_triage.ieee118_dataset --output-dir output/classifier_triage_20261011/ieee118_pipeline_corpus --workers 14 --stages waveform
python scripts/build_balanced_corpus.py --system case118 --output-dir output/classifier_triage_20261011/ieee118_balanced_corpus_dev --seed 20261011 --counts '{"no_error": 60, "measurement_error": 240, "parameter_error": 240}' --load-scale-min 0.80 --load-scale-max 1.25
python -m research.classifier_triage.ieee118_suite --balanced-corpus output/classifier_triage_20261011/ieee118_balanced_corpus_dev --waveform-corpus output/classifier_triage_20261011/ieee118_pipeline_corpus/pipeline_corpus --output-dir output/classifier_triage_20261011/suite_ieee118
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261011/suite_ieee118/development.json --profile classifier_gated_diagnostics --triage-model psse_env/oracle/models/triage_gnn_joint_20261011 --progress-basis chi_square_excess --output-dir output/classifier_triage_20261011/e2e_ieee118_excess
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261011/suite_ieee118/development.json --profile classifier_gated_diagnostics --triage-model psse_env/oracle/models/triage_gnn_joint_20261011 --progress-basis chi_square_statistic --output-dir output/classifier_triage_20261011/e2e_ieee118_statistic
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261011/suite_ieee118/development.json --profile classifier_gated_diagnostics --triage-model psse_env/oracle/models/triage_gnn_joint_20261011 --progress-basis chi_square_excess --families harmonic three_phase_unbalance hif measurement+hif --output-dir output/classifier_triage_20261011/e2e_ieee118_excess_waveform_fix   # after the noise-floor fix
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261004/suites/development.json --profile classifier_gated_diagnostics --families harmonic three_phase_unbalance hif measurement+hif --output-dir output/classifier_triage_20261011/e2e_ieee14_waveform_fix   # IEEE 14 unchanged: 56/56
# the cell: research/hpc/full_pipeline_20260907/overrides/ieee118_transfer_20261011.env (deploy comment inside)
```

Files: `research/classifier_triage/ieee118_dataset.py`,
`research/classifier_triage/transfer_ieee118.py`,
`research/classifier_triage/transfer_ieee118_to_14.py`,
`research/classifier_triage/tests/test_transfer_ieee118.py`,
`three_phase_model/runtime.py` (acceptance fallback),
`tests/test_ieee118_three_phase_model.py` (its regression test),
`Transmission/generate_hse_traces.py` (harmonic reference bus).
