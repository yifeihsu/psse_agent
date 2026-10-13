# Effect-targeted fault magnitude (2026-10-12)

## Why

A root is kept when the balanced WLS alarms, and a triage classifier is
asked to tell families apart by the pattern of its residuals and Lagrange
multipliers. Yet every family's magnitude was drawn in physical units (a
source-bus THD of 10 to 20%, a fault resistance in ohm bands, a meter error of
5 to 15 sigma, a line factor), and the same recipe lands at very different
detector magnitudes on different networks. Measured as the chi-square excess
over the degrees of freedom divided by the alarm threshold's own excess (so 1
is "just alarming" on any network), the development suites of October 11 gave:

| family | IEEE 14 | IEEE 118 |
| --- | --- | --- |
| harmonic | 13.7 | 3.8 |
| hif | 2.2 | 4.2 |
| measurement | 4.3 | 1.1 |
| measurement+hif | 16.1 | 4.1 |
| multi_measurement | 11.6 | 4.6 |
| parameter | 28.0 | 2.9 |
| three_phase_unbalance | 5.8 | 8.5 |

On IEEE 14 a harmonic root is the loudest family and an HIF the quietest; on
IEEE 118 the order differs. Size is therefore a family cue, and a different
one on each network: the IEEE 14 gate's HIF recall on IEEE 118 depended on the
strength range, and the IEEE 14 student's spectra request keyed on loudness
and breadth and never fired on IEEE 118. The remedy the user asked for on
October 12: draw every magnitude as the effect the detector sees, in one band
shared by every family and system, so the classifier is left with the shape.

## The sampler (`research/classifier_triage/effect_sampler.py`)

- Magnitude metric: `effect_ratio = (J - dof) / (chi2.ppf(0.99, dof) - dof)`.
  The threshold's excess grows like the square root of the degrees of freedom
  (32 on IEEE 14, 97 on IEEE 118), so the ratio is the network-normalized
  size. The local test is not needed as a second knob: a single meter at ratio
  1.5 already stands 7 sigma out on IEEE 14 and 12 on IEEE 118.
- Band: log-uniform between 1.5 and 20, one draw per root.
- Search: the disturbed telemetry is `reference + k * deviation + noise`, with
  `reference` the same operating point without the disturbance, `deviation`
  the noiseless change the fault makes, one fixed sensor-noise draw, and `k`
  found by bisection on the log scale (8 to 13 WLS solves; 0.2 s on IEEE 14,
  8 s on IEEE 118). A bound where the WLS diverges is pulled inwards until it
  converges; a band the bracket cannot reach rejects the draw.
- Implied physical magnitude, recorded next to the drawn one: a meter
  amplitude scales with `k` (exact); a harmonic source's THD with `sqrt(k)`,
  since the transducer errors are quadratic in the distortion (exact to second
  order); a fault's current and power with `k` and its resistance with `1/k`
  (first order, Thevenin impedance small against the resistance); an
  unbalance's phase split deviates from equal shares by `k` times the drawn
  deviation (first order); a line factor's logarithm scales with `k`, and the
  true network is re-solved at the implied factor, so parameter rows stay
  exact.
- Phasors of the three-phase families are re-magnituded consistently
  (`clean + k * (disturbed - clean)` per phasor), so the PMU rungs see the
  implied fault.

## Where it is applied

- `psse_env/providers/balanced_corpus.py` (`effect_band=`): meter rows scaled
  exactly; parameter rows re-solved by the true network's OPF at the implied
  factors (bounded so factors stay within 0.02 to 50); unreachable rows
  rejected and counted.
- `research/classifier_triage/ieee118_dataset.py`: any registered system
  (`--system`), `--effect-band`, waveform rows taken from corpora
  (`--hif-corpus`, `--unbalance-corpus`; rows cycle with fresh targets and
  noise when the plan exceeds them) instead of the three-phase replays,
  harmonic synthesis scaled, HIF-plus-meter overlays and flow-pair mimics
  targeted as increments from the band, admission = the operator alarm and
  the root's own ratio in the band (15% slack). Under a band the generator
  admits on the plain alarm and the ten-sigma meter floor is lifted; the
  measurement+parameter family is left out because its overlay follows the
  generator's own law. The regenerated HIF and unbalance corpora are written
  in the pipeline's row schema (`pipeline_corpus/`).
- `psse_env/providers/scenario_generator.py` (`effect_band=`): the harmonic
  roots a generator synthesizes are scaled the same way; the corpus-drawn
  families carry the band from their corpora.
- `research/classifier_triage/features.py`: view `residual_v3`, residual_v2
  with the absolute excess replaced by the signed log of the effect ratio.
- `research/classifier_triage/transfer_effect.py`: per-dataset models with
  thresholds at the IEEE 14 export's recall target on their own calibration
  split, scored in-domain and zero-shot on the other dataset's test split
  (own threshold, and a threshold re-chosen on the target), plus the joint
  model.

## Datasets

Seed 20261012, band 1.5 to 20, splits by physical parent (60/20/20).

- `output/classifier_triage_20261012/ieee14_effect`: the four IEEE 14 HIF
  corpora (131 windows) and the unbalance corpus (162 windows) cycled to 400
  and 300 draws, 300 harmonic syntheses, 8 balanced shards (16 meter, 16
  multi-meter, 16 parameter roots each from a fresh 24/72/48 corpus), 100
  flow-pair mimics per variant, 2 healthy replicas per window.
- `output/classifier_triage_20261012/ieee118_effect`: the IEEE 118 pipeline
  corpora (665 HIF, 289 unbalance windows) cycled to 700 and 450 draws, 300
  harmonic syntheses, 12 balanced shards, same per-shard plan.

## Results

Roots per dataset and their effect ratios (median [min, max] over the roots in the band, test and train together):

| family | IEEE 14 n | IEEE 14 ratio | IEEE 118 n | IEEE 118 ratio |
| --- | --- | --- | --- | --- |
| harmonic | 300 | 4.8 [1.5, 19.8] | 300 | 4.9 [1.5, 20.1] |
| hif | 400 | 5.8 [1.4, 20.5] | 700 | 5.8 [1.5, 20.2] |
| measurement | 128 | 4.7 [1.5, 19.6] | 192 | 4.8 [1.5, 20.6] |
| measurement+hif | 353 | 11.9 [3.2, 22.6] | 621 | 12.0 [3.3, 22.9] |
| multi_measurement | 128 | 7.5 [2.1, 20.3] | 192 | 6.0 [1.5, 20.8] |
| parameter | 128 | 5.3 [1.4, 21.0] | 192 | 4.3 [1.4, 20.2] |
| three_phase_unbalance | 300 | 4.8 [1.5, 20.4] | 450 | 5.2 [1.5, 19.9] |

### View `residual_v3` (band 1.5 to 20.0, 5 seeds, thresholds at the IEEE 14 export's recall target on the training network's calibration split)

| trained on | scored on | recall | unneeded requests | AUC | at a threshold re-chosen on the target: recall / unneeded |
| --- | --- | --- | --- | --- | --- |
| case118 | case118 | 98.7% | 0.0% | 100.0% | - |
| case118 | case14 | 77.6% | 0.0% | 96.5% | 99.2% / 73.9% (threshold 0.0000) |
| case14 | case118 | 66.6% | 0.5% | 97.9% | 98.7% / 36.2% (threshold 0.0012) |
| case14 | case14 | 99.2% | 0.9% | 99.9% | - |
| joint | case118 | 98.7% | 0.9% | 100.0% | - |
| joint | case14 | 99.6% | 0.9% | 100.0% | - |

Requests by family at the own threshold, trained on case118 and scored on case14: harmonic 44/55, healthy_window 0/7, hif 69/75, measurement 0/21, measurement+hif 62/68, mimic_flow_pair_opposite_sign 0/22, mimic_flow_pair_same_sign 0/15, multi_measurement 0/17, parameter 0/33, three_phase_unbalance 23/57.
Requests by family at the own threshold, trained on case14 and scored on case118: harmonic 12/55, healthy_window 0/55, hif 122/130, measurement 0/35, measurement+hif 105/112, mimic_flow_pair_opposite_sign 0/22, mimic_flow_pair_same_sign 0/28, multi_measurement 1/43, parameter 0/38, three_phase_unbalance 14/83.

### View `residual_v2` (band 1.5 to 20.0, 5 seeds, thresholds at the IEEE 14 export's recall target on the training network's calibration split)

| trained on | scored on | recall | unneeded requests | AUC | at a threshold re-chosen on the target: recall / unneeded |
| --- | --- | --- | --- | --- | --- |
| case118 | case118 | 98.4% | 0.0% | 100.0% | - |
| case118 | case14 | 75.3% | 0.0% | 97.0% | 98.4% / 55.7% (threshold 0.0000) |
| case14 | case118 | 67.1% | 0.0% | 97.9% | 98.7% / 41.2% (threshold 0.0031) |
| case14 | case14 | 98.8% | 1.7% | 99.7% | - |
| joint | case118 | 98.4% | 0.0% | 100.0% | - |
| joint | case14 | 99.6% | 0.9% | 99.9% | - |

Requests by family at the own threshold, trained on case118 and scored on case14: harmonic 40/55, healthy_window 0/7, hif 71/75, measurement 0/21, measurement+hif 62/68, mimic_flow_pair_opposite_sign 0/22, mimic_flow_pair_same_sign 0/15, multi_measurement 0/17, parameter 0/33, three_phase_unbalance 19/57.
Requests by family at the own threshold, trained on case14 and scored on case118: harmonic 15/55, healthy_window 0/55, hif 122/130, measurement 0/35, measurement+hif 104/112, mimic_flow_pair_opposite_sign 0/22, mimic_flow_pair_same_sign 0/28, multi_measurement 0/43, parameter 0/38, three_phase_unbalance 14/83.


### Reading

Against the physical-magnitude study (IEEE 14 export zero-shot on IEEE 118:
recall 56.7%, HIF 79%, HIF-plus-meter 67%, harmonic 41%, unbalance 10%;
IEEE 118 model zero-shot on IEEE 14: 92.3%, harmonic 50/50, HIF 25/25,
unbalance 22/30):

- **Size no longer carries the family.** Within one band every family sits at
  a median ratio of 4 to 7 on both networks, mimics and all; a classifier can
  only read the shape.  Unneeded requests stay at 0 to 1% in every direction.
- **HIF transfers both ways once sizes match.** Forward, the IEEE 14 model now
  requests phasors on 122 of 130 IEEE 118 HIF roots (94%, against 79% when
  IEEE 118's faults were two to five times stronger than anything in the IEEE
  14 training range) and on 105 of 112 HIF-plus-meter roots; reverse, the IEEE
  118 model takes 69 of 75 IEEE 14 HIFs.  The fault line's terminal footprint
  is the same shape on any network.
- **Harmonic and unbalance do not, in either direction.** Forward, harmonic 12
  of 55 and unbalance 14 of 83; reverse, harmonic 44 of 55 and unbalance 23 of
  57.  The reverse direction is worse than before (92% overall became 78%)
  because the old result leaned on loudness: on IEEE 14 the waveform families
  were the loud ones, and the IEEE 118 model's "loud means waveform" habit paid
  off there.  With loudness removed, what is left is the shape, and the shape
  of these two families is network-bound.  With sizes matched, an IEEE 14
  harmonic root flags 8 channels above 3 sigma against 19 on IEEE 118, its top
  five residuals hold 55% of the excess against 35%, and an IEEE 14 unbalance
  leaves the multipliers quiet (largest multiplier 0.8 of the largest
  residual) where an IEEE 118 unbalance excites them (1.6).  The spread of a
  distortion or an unbalance over channels follows the network's size and
  coupling; the model has no network-free representation of it.
- **The misses score near zero.** Re-choosing the threshold on the target
  recovers recall only by flagging 36% (forward) or 74% (reverse) of the
  balanced roots.  The AUCs of 97 to 98% are carried by the HIF families; the
  harmonic and unbalance misses are not borderline.
- **The joint model holds both networks at 98.4 to 99.6% recall with 0 to 0.9%
  unneeded requests**, as before.  The two views (absolute excess, relative
  ratio) behave alike, which confirms that size was not what the model keyed on
  for these two families; the per-channel pattern was.

What follows for the student: the band is the right data contract (it removes
a confound and fixes the HIF strength-range dependence at the gate), but it
does not by itself make the waveform cues transferable; multi-network
training, or a shape representation that is explicit about how a pattern is
spread relative to the network, is still needed for harmonics and unbalance.
