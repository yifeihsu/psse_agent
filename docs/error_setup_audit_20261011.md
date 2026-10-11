# Are the error setups of IEEE 14 and IEEE 118 the same distribution? (2026-10-11)

Question: the triage GNN transfers badly from IEEE 14 to IEEE 118 (58% recall)
and imperfectly back (92%). If the two sets draw their errors differently, no
model can be expected to transfer. This audit compares, family by family, what
the two sets draw (the error parameters), what the balanced WLS then sees (the
alarm signature), and how the sets are composed. Tool:
`python -m research.classifier_triage.error_setup_audit`; outputs under
`output/classifier_triage_20261011/error_setup_audit` (`error_setup_audit.md`
has every quantile and KS statistic).

KS is the two-sample Kolmogorov-Smirnov statistic between the networks: 0 is the
same distribution, 1 disjoint. Quantiles are 10 / 50 / 90%.

## Verdict

The recipes are the same on paper for every family, and the parts that are
expressed in network-free units match in the data: meter errors in sigmas,
parameter factors, harmonic THD, unbalance phase fractions, HIF phase and split,
load scale and dispatch, SCADA noise. Four things are not consistent, and three
of them are setup choices rather than physics:

1. **HIF strength is drawn in ohms.** IEEE 14 training HIFs sit on 7 lines at
   69 kV (47.6 ohm base); IEEE 118's on 175 lines at 138 kV (190 ohm) and
   345 kV (1,190 ohm). The same 100-1,000 ohm bands give 2.1-4.4 pu on IEEE 14
   and 0.6-3.7 pu on IEEE 118 (10-90%), fault current 0.30 vs 0.61 pu, KS 0.6.
2. **The admission rule's chi-square branch scales with the network.** The
   rule is the same (J > 1.25 x chi2(0.99, dof) or max |r| >= 5), but the
   excess objective it demands is 67.5 on IEEE 14 and 340 on IEEE 118. On IEEE
   118 the chi-square branch admits nothing on its own: every admitted root is
   a max |r| >= 5 root. The chi-square *feature* log(J / threshold) is
   therefore on different scales: an identical 12-sigma meter error reads 0.48
   on IEEE 14 and 0.02 on IEEE 118 (KS 0.96).
3. **The two sets are composed differently.** 35% of IEEE 14 rows need
   phasors against 56% on IEEE 118; IEEE 118 has no topology family; IEEE 14
   healthy alarms are diffuse chi-square alarms (59% chi-square-only, median
   max |r| 3.6) while IEEE 118's are single-channel spikes (86% max |r|-only,
   median 4.15); IEEE 118 draws 2-4 noise replicas per healthy window, IEEE
   14 one.
4. **At matched physical strength the signatures still differ, and that is
   the network, not the setup.** A 3 pu HIF or a 10 MW negative-sequence load
   lights up two to five times more channels and assets on IEEE 118, with a
   lower share of its energy in the top two channels (unbalance: 0.61 vs 0.20,
   KS 0.93). Part of this is a noise floor: by chance about 3 channels exceed
   3 sigma on 1,098 channels against 0.3 on 122.

Everything the two three-phase models assume is the same: diagonal phase
matrices without mutual coupling (Z0 = Z1), single-phase constant-PQ loads,
Model=3 voltage-regulated generators with reactive limits, a voltage source at
the slack, Vm = phase A, total three-phase powers, shunts kept in Ybus, and a
noiseless healthy reference that the balanced WLS fits exactly (J = 0 on both).

## Recipe, side by side

| Dimension | IEEE 14 (study rows) | IEEE 118 set | Same? |
| --- | --- | --- | --- |
| Operating point | loads x U(0.80, 1.25), AC-OPF dispatch | same | yes |
| SCADA noise | 1e-3 on Vm, 1e-2 on power | same | yes |
| HIF location | 7 lines, all 69 kV (the `69kv` stratum of 16 same-voltage lines) | 175 same-voltage lines (165 at 138 kV, 10 at 345 kV), 169 used | **no** (coverage) |
| HIF resistance | 100-200 / 200-500 / 500-1000 ohm, equal weight | same | same ohms, **different pu** |
| HIF admitted R | 91% in the 100-200 band (161 ohm median) | uniform over the bands (341 ohm median) | **no** (selection) |
| HIF split, phase | U(0.25, 0.75); A, B, C uniform | same (KS 0.05) | yes |
| HIF with a bad meter | one power meter, max(U(0.10, 0.30), 10 sigma x U(1, 1.5)) | same (sigma multiples KS 0.12) | yes |
| Unbalance split | Dirichlet(3, 3, 3) at one non-slack load bus, total kept | same (LUF KS 0.23) | yes |
| Unbalance buses | 11 eligible (9 used), 25 MW median load | 99 eligible (75 used), 45 MW median | **partly** (bus size) |
| Harmonic | OPF point, source bus with load, THD U(0.10, 0.20), orders 5-19 | same (THD KS 0.11) | yes |
| Meter errors | one channel, U(5, 15) sigma lifted to >= 10 sigma x U(1, 1.5) | same (KS 0.12; channel mix equal) | yes |
| Multi-meter | 2-5 meters of one channel type | same (count KS 0.04) | yes |
| Parameter | one line, R / X / both, factor U(0.1, 0.5) or U(2, 5) | same (KS 0.07-0.11) | yes |
| Mimics | both flow meters of one line, 10-15 sigma | same (KS 0.10) | yes |
| Admission | J > 1.25 thr or max abs r >= 5 | same rule | **no** (effect) |
| Healthy windows | 1 draw per corpus window and HIF control, kept on alarm | 2-4 draws per window, kept on alarm | **no** |
| Topology | 250 + 250 roots (node-breaker model) | none | **no** |
| Phase coupling, loads, generators | diagonal, 1-phase PQ, Model=3 PV, Vsource slack | same | yes |

## What the WLS sees

**Network-free units match; network-scaled ones do not.** Per family, the
features with KS >= 0.5 (the full table is in the audit output):

| Family | Matches (KS) | Differs (KS) |
| --- | --- | --- |
| HIF | phase 0.05, split 0.05, Vm share 0.31 | kV 1.0, assets 0.85, channels 0.81, reactive share 0.80, multiplier 0.73, excess J 0.72, fault current 0.62, R pu 0.61, top-2 share 0.61, max abs r 0.60 |
| HIF + bad meter | meter sigmas 0.12, max abs r 0.22 | assets 0.80, channels 0.73, R pu 0.61, top-2 0.59, excess 0.54, log ratio 0.50 |
| Unbalance | max abs r 0.11, log ratio 0.14, fractions 0.21-0.23, neg-seq MW 0.31 | top-2 share 0.92, assets 0.77, channels 0.67, excess 0.55, multiplier 0.51 |
| Harmonic | THD 0.11, excess 0.13, max abs r 0.12, multiplier 0.09, channels 0.23 | log ratio 0.53 |
| Measurement | sigmas 0.12, excess 0.12, max abs r 0.11, multiplier 0.11 | log ratio 0.96, assets 0.62, channels 0.55, top-2 0.53 |
| Multi-meter | everything else 0.04-0.30 | log ratio 0.98 |
| Parameter | factors 0.07-0.11, excess 0.12, multiplier 0.11, max abs r 0.14 | none (log ratio 0.42) |
| Meter + parameter | factors, sigmas, excess 0.20 | log ratio 0.54 |
| Healthy alarms | | assets 0.70, channels 0.68, top-2 0.55, log ratio 0.53 |
| Mimics | sigmas 0.10, excess 0.10, max abs r 0.14 | log ratio 1.0, assets 0.56-0.76, channels 0.52-0.76 |

Three patterns run through the table:

- `log(J / threshold)` differs on every family, including those whose excess
  objective J - dof matches (meters, parameters, harmonics, mimics). The same
  injected residual energy is a smaller multiple of a threshold that grows
  with the degrees of freedom. It is the only GNN input on a network scale.
- `channels > 3 sigma` and `assets` are offset by the chance exceedances of a
  larger channel count (healthy alarms: median 1 vs 4 channels), so every
  count-type signature is shifted up on IEEE 118 even when the error is the
  same.
- The waveform families differ beyond these two effects: HIF and unbalance
  footprints are spread across far more assets on IEEE 118.

**Matched strength isolates the network effect.** Restricting both sets to the
same physical error:

| Selection | Rows (14 / 118) | Still matching | Still differing |
| --- | --- | --- | --- |
| HIF at 2.1-5.3 pu (138 kV only on IEEE 118) | 125 / 250 | R pu 0.12, fault current 0.16, max abs r 0.14 (7.1 vs 7.5), excess 0.40 (118 vs 222) | assets 0.70 (4 vs 9), channels 0.63 (5 vs 11), reactive share 0.69 (0 vs 0.38), multiplier 0.54 (3.0 vs 5.3), top-2 0.50 |
| Unbalance at 5-30 MW negative sequence | 97 / 201 | neg-seq MW 0.23, max abs r 0.04 (9.06 vs 8.99), log ratio 0.18 | top-2 share 0.93 (0.61 vs 0.20), assets 0.74 (5 vs 25), channels 0.66 (8 vs 33), excess 0.58 (189 vs 746), Vm share 0.53, multiplier 0.53 |
| Harmonic at THD 0.12-0.18, source 5-60 MW | 113 / 227 | THD 0.10, channels 0.18, assets 0.25, top-2 0.27, excess 0.28 | log ratio 0.69, reactive share 0.52 |
| One bad power meter of 11-14 sigma | 130 / 135 | sigmas 0.15, excess 0.14, max abs r 0.15, multiplier 0.16 | log ratio 0.96, channels 0.64 (3 vs 5), assets 0.68 (2 vs 5), top-2 0.62 |

So: once the HIF strength is matched, the alarm statistic the rule reads (max
|r|) is the same on both networks, but the footprint is not. Unbalance is the
extreme case: the same negative-sequence load gives the same max |r| yet four
times the residual energy, spread over five times the assets. This is not a
recipe choice. The faulted lines are as close to regulating generators on both
networks (82% vs 81% adjacent to a generator bus), the phase coupling is the
same, and the loads and generators are modelled the same way; what differs is
the meshing of a 118-bus 138 kV network against a 5-bus 69 kV ring. A model
must see such footprints in training to recognise them.

## The admission rule, in numbers

| | IEEE 14 | IEEE 118 |
| --- | --- | --- |
| chi-square threshold at 1% | 129.97 (dof 95) | 962.58 (dof 863) |
| excess J - dof needed at margin 1.25 | 67.5 | 340.2 |
| excess needed at margin 1.0 (healthy windows) | 35.0 | 99.6 |
| roots admitted by chi-square only | HIF 1%, unbalance 4%, harmonic 2%, parameter 8% | 0% in every family |
| roots admitted by max abs r only | HIF 18%, harmonic 5%, meter 9%, parameter 4% | HIF 31%, harmonic 48%, meter 97%, parameter 38% |

On IEEE 118 the pipeline rule and "max |r| >= 5" admit exactly the same draws
(633/633 HIF, 233/233 unbalance, 439/439 harmonic). An IEEE 14-equivalent
excess rule (J - dof >= 67.5 or max |r| >= 5) would admit 35 more HIFs (+6%),
86 more unbalances (+37%) and 31 more harmonics (+7%): the diffuse, energetic
roots the IEEE 118 set lacks. In the other direction almost every IEEE 14 root
survives the IEEE 118-equivalent rule (HIF 130/131, unbalance 155/162,
parameter 231/250), so both admitted populations are, in effect, max |r| >= 5
populations, and the chi-square branch mostly changes the *feature*, not the
selection.

## What to change for a transferable setup

1. **Draw HIF strength in per unit of the local base** (or by fault current),
   with the same bands on every network, and on every same-voltage line.
   Keep ohms only as a derived label. (IEEE 14's 69 kV stratum maps 100-1,000
   ohm to 2-21 pu, IEEE 118's lines to 0.08-5.3 pu: the bands do not overlap
   below 2.1 pu.)
2. **Admit by network-free statistics** (done, `excess_energy` rule): max |r|
   >= 5 (already the effective rule) and an excess-energy floor J - dof >= 70,
   instead of a multiple of the dof-dependent chi-square threshold. Healthy
   windows keep the operator alarm: the excess floor is a margin on true
   errors, and the negatives must be the states the deployed detector flags.
3. **Replace the chi-square-ratio input** (done, view `residual_v2`) by the
   excess energy on a network-free scale, log1p((J - dof) / 70), and the >3 /
   >4 sigma counts in excess of chance (0.0027 n and 6.3e-5 n). The node and
   edge flags are per channel and need no change.
4. **Match the composition**: the same positive share, the same families
   (topology on IEEE 118, or none on IEEE 14 for transfer training), the same
   number of noise replicas per healthy window.
5. **Stratify unbalance by negative-sequence load** (LUF x bus MW) rather than
   by fractions alone, since IEEE 118 buses carry about twice the load.
6. **Accept the network effect and train across sizes.** The footprint of a
   matched HIF or unbalance is a property of the network; IEEE 57 rows of
   every family would give a third size between 14 and 118.

Items 1-3 are recipe and feature changes in `ieee118_dataset.py`,
`scenario_generator.py` and `features.py`; item 4 is dataset assembly; none
needs new physics. Items 2 and 3 were applied the same day (next section).

## Fixes 2 and 3 applied (2026-10-11)

**Fix 2, the admission rule.** `psse_env/wls_admission.py` holds the operator
alarm and both admission rules as pure functions. The generator takes
`admission_rule="excess_energy"` (default `chi_square_margin`, so every
existing pipeline run is unchanged): a root is admitted when the operator
alarm fires and max |r| >= 5 or J - dof >= 70. The IEEE 118 builder uses it
for its own roots and for the generator's balanced families; healthy windows
and mimics keep the operator alarm, because that is the population the gate
meets in deployment (the excess floor is a margin on true errors, not a
detector). The rebuilt set (`output/classifier_triage_20261011/ieee118_v2`)
holds 3,106 roots against 2,961: the excess branch alone admits 32 more HIFs,
56 more unbalances (+24%), 26 more harmonics and 13 more parameter roots,
the diffuse roots the chi-square margin had excluded; alarm-only draws fall
from 173 to 59. On IEEE 14 the rule drops 2 of 2,148 roots, so the IEEE 14
study rows are re-filtered at load time instead of regenerated
(`transfer_ieee118 --admission-rule excess_energy`).

**Fix 3, the global inputs.** `features.py` gains the view `residual_v2`:
the same node and edge features, with the five globals replaced by the
signed log of (J - dof) / 70, log1p of the >3 sigma and >4 sigma counts in
excess of chance (0.0027 n and 6.3e-5 n live channels), and the unchanged
largest residual and multiplier. Checkpoints carry their view, the runtime
accepts both residual views, and the tracked IEEE 14 export stays a legacy
`residual` model, so the deployed gate is untouched.

**Effect on transfer: none.** The same GNN was trained (five seeds) on IEEE
14 alone, IEEE 118 alone and both, under each combination, and scored
zero-shot on the other network's test split at the training network's own
calibration threshold:

| Setup | Trained on 14, scored on 118 | Trained on 118, scored on 14 | Joint on 118 | Joint on 14 |
| --- | --- | --- | --- | --- |
| legacy (old set, legacy view) | 56.7% / 0.3% (the export) | 92.3% / 1.2% | 97.2% / 0.7% | 100% / 0.6% |
| fix 3 alone (old set, v2 view) | 51.1% / 0.3% | 91.5% / 1.2% | 98.1% / 0.0% | 100% / 0.9% |
| fix 2 alone (new set, legacy view) | 53.3% / 0.7% | 94.6% / 1.5% | 98.0% / 1.3% | 98.4% / 0.6% |
| fixes 2 + 3 (new set, v2 view) | 44.1% / 0.3% | 93.8% / 0.6% | 97.4% / 1.3% | 98.4% / 0.6% |

Recall / unneeded requests. The forward numbers move with the IEEE 14
threshold each training produced (0.24 to 0.41) more than with the recipe:
the fixes-2+3 model at the export's threshold 0.278 recalls 50.3%, at 0.10
65% (0.7% unneeded), at 0.03 76% (1.7%). The AUCs are the same throughout
(97.0 to 97.7% forward, 98.5 to 98.7% reverse). By family the picture is
unchanged: forward, HIF 70-75%, HIF with a bad meter 59-66%, harmonic
11-34%, unbalance 2-9%; reverse, every family above 92% except unbalance
(77-83%). Splitting the forward misses by admission branch shows the new
excess-only roots are not the cause (recall on the old population 45.6%,
on all 44.1%).

So the two inconsistencies are removed, and the transfer gap stays where the
audit located it: in the HIF strength distribution (fix 1) and in the
network's own footprints, which only training data from the target network,
or from several network sizes, can supply. The joint model remains the
working answer (97-98% on both networks at 0.6-1.3% unneeded requests).
