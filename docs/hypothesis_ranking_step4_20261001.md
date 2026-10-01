# Hypothesis ranking, step 4: a learned ranker against the physics screen (2026-10-01)

Step 4 of `docs/hypothesis_ranking_plan_20260930.md`, with the scope step 3
narrowed it to: on IEEE 14 the screen's ranking already matches the multiplier
ranking and the ledger expert follows it at lower cost, so the learned ranker
is judged on the three discriminations the rules leave open (the same-sign
flow-meter pair against a true HIF, a voltage-meter error against an
unbalance, the adjacent-line ambiguity) and on whether it can gate phasor
acquisition better than the shipped rule. Code
`research/hypothesis_ranking/ranker.py`; results
`output/hypothesis_ranking_20260930/ranker/` (gitignored). Research only.

## 1. Data, features, protocol

The step 1 states re-analyzed by the step 2 screen
(`output/hypothesis_ranking_20260930/ieee14_v3`): alarmed roots of every
family, the 69 alarmed healthy windows, the 200 flow-meter mimics, and the
truth-corrected child states (training only). Splits are the dataset's
parent splits: 2,277 training rows (781 of them children), 458 calibration
and 463 test rows (roots, healthy alarms and mimics only). 94 features, all
read from what the policy sees: the WLS ledger (chi-square ratio, residual and
multiplier magnitudes and ranks, channel types of the top residuals, the
dominance tags, a flag for two flow meters of one branch among the top
residuals and whether they agree in sign), the screen's first-round class
scores, margin, winner and best targets, the accepted sequence, the class
tests, the final solve, and the ranked-target gaps. Labels are the truth's
family presence; `needs_aux` is HIF, unbalance or harmonic (under C2 a
harmonic root reaches its spectra only through balanced phasors). Models:
a gradient-boosted classifier and a standardized logistic regression per
target, Platt-scaled on the calibration split; every number below is on the
test split with a 95% parent-bootstrap interval.

## 2. Results

**Discrimination is high for every family** (gradient-boosted AUC on the
test split): measurement 99.9%, parameter 99.6%, topology 100%, HIF 99.7%,
unbalance 98.3%, harmonic 99.1%, `needs_aux` 99.8%; the logistic model is
within a point on each. Calibration after Platt scaling: expected calibration
error 0.002 to 0.027.

**1. Acquisition gate.** Rule v3 (HIF won, voltage-meter pick, or
unexplained) against the learned `needs_aux` score, thresholds set on the
calibration split:

| operating point | recall on roots that need an auxiliary stream (130) | acquisitions on roots that need none (333) |
| --- | --- | --- |
| rule v3 | 98.5% [95.9, 100] | 12.0% [8.5, 15.7] |
| learned, threshold at the rule's calibration false rate | 100% [100, 100] | 15.3% [11.7, 19.1] |
| learned, threshold at the rule's calibration recall | 96.9% [93.4, 99.3] | 1.2% [0.3, 2.5] |

The learned score does not dominate the rule at either operating point: it
buys a ten-point drop in unnecessary acquisitions (paired difference
-10.8% [-14.2, -7.4]) for about 1.5 points of recall (2 of 130 roots, one
harmonic and one unbalance), or picks up the rule's two misses at a slightly
higher false rate. The score ranks almost perfectly (AUC 99.8%), so a
threshold between the two exists, but choosing it on the calibration split is
exactly what was done here, and the gap between the calibration and test
false rates of the rule itself (15.9% against 12.0%) shows how much the small
negative sets move. The permutation importance is physically sensible: the
number of voltage channels set aside, the second residual's magnitude, the
final solve's objective ratio, and whether the screen explained the alarm.

**2. HIF against the same-sign flow-meter pair.** At the screen's HIF recall
(threshold from the calibration split), the learned HIF score flags none of
the 16 same-sign mimics in the test split where the screen flags 37.5%
(paired difference -37.5% [-61.1, -15.8]; AUC of HIF roots against mimics
98.8% [95.6, 100]), at a cost of one HIF root in 50 (96.0% [89.6, 100]
against the screen's 98.0% [92.9, 100]). This is the one discrimination the
plan named where learning adds something the refits cannot: the pair of
flow meters leaves a different residual pattern from a shunt, and the model
reads it.

**3. Unbalance against a voltage-meter error.** Among the 38 test roots with
a voltage-meter pick that are either an unbalance (29) or a true
voltage-meter fault or healthy alarm (9), the learned unbalance score has
AUC 92.7% [83.6, 99.1]. So the two are not indistinguishable on balanced
evidence after all: the pattern of voltage channels across neighbouring
buses carries information. The negative class is nine roots, the interval is
wide, and the acquisition saved at 95% unbalance recall (55.6% [16.7, 88.9])
is descriptive only.

**4. Adjacent-line ambiguity.** On the 95 test parameter roots whose true
line is one of the multiplier ranking's top two, taking the top line is right
94.7% of the time, the screen's top line the same, and a pairwise logistic
model on the two candidates' multipliers, refit objectives and adjacency the
same (difference 0.0 [0.0, 0.0]). Nothing in the balanced evidence separates
the misranked cases; this remains the identifiability ceiling.

**5. Transfer to IEEE 57** (models and thresholds unchanged): the
`needs_aux` score ranks the 184 alarmed study roots at AUC 84.5%
[76.2, 93.8], but the IEEE-14 threshold is useless there (recall 93.7% at a
65.4% false rate against the rule's 63.9% and 11.5%); the HIF score does not
transfer (AUC 65.9%, recall 4.5%). The features carry over, the operating
points do not, and the IEEE 57 HIF roots look like nothing the IEEE 14
screen flags.

## 3. Decision

Against the plan's criterion (replace the screen's scores in the ledger only
when the ranker improves the flow-pair confusion or coverage with an interval
that excludes zero at the same false-acquisition rate): the flow-pair result
qualifies, the acquisition result trades recall for false rate rather than
improving one at the other held fixed, and the transfer result says the
operating points are network-specific.

Recommendation: keep the physics rule as the admission gate (the process
oracle and the provider), and use the learned scores as an ordering signal
in the ledger expert. Concretely, when the rule admits a phasor request on a
root whose learned `needs_aux` probability is low (a same-sign flow pair, a
multi-meter root), the ledger tries the leading balanced hypothesis before
spending the acquisition; when the probability is high it acquires at once.
This keeps every root the rule reaches reachable (no recall is given up),
spends the saved acquisitions on the roots where the model is confident, and
puts the classifier where the plan wanted it: recommending, never bypassing
the controller. Implementing that ordering, measuring it with the step 3
paired harness, and a calibrated retraining on IEEE 57 data are the next
items; the GNN screen was not retrained here (the gradient-boosted model
on the screen's own outputs already reaches 99.8% AUC, leaving nothing for a
graph model to add on this network).

## 4. Reproduce

```bash
LOKY_MAX_CPU_COUNT=8 python -m research.hypothesis_ranking.ranker --dataset-dir output/hypothesis_ranking_20260930/ieee14_v3 --transfer-dir output/hypothesis_ranking_20260930/ieee57_v3 --output-dir output/hypothesis_ranking_20260930/ranker
```

About 30 s. The report, the metrics with every interval, and the test-split
scores per root are written beside each other.
