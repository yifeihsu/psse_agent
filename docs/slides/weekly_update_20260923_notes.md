# Presenter notes — weekly update, September 23, 2026

Suggested pacing: slides 2–10 summarize the experimental changes and results;
slides 11–14 explain HIF compensation; slides 15–17 cover limitations and next
experiments. Slides 18–19 are backup detail.

## 1. Title

This update covers physical scenario construction, the completed second DAgger
round, and HIF-conditioned measurement repair. Keep the completed experiments
separate from the new SCADA-only protocol throughout the presentation.

## 2. This week's changes

There are three distinct changes: the data-generating physics, training the R2
policy, and changing the runtime/audit used to evaluate frozen weights. R1 to
R2 adds one success. The larger subsequent +16 uses unchanged R2 weights and
comes from eight actual meter repairs plus eight healthy-control audit fixes.
Current SCADA-only performance has not been measured end to end.

## 3. Noise, WLS and audit consistency

Noise is part of the acquired data, so healthy observations must not be compared
as though the model should reconstruct the exact noiseless vector. Covariance
must describe the added noise. For topology projections, propagate the sensor
covariance; exact structural-zero constraints are handled separately. The
chi-square and normalized-residual tests have different roles. The OR rule
does not inherit a calibrated overall one-percent false-alarm probability.

## 4. Fault injection settings

The main corpus intentionally includes gross observable errors. The meter
floor is at least ten sigma; it is not a universal claim that all errors are
sampled uniformly from ten to fifteen sigma. Parameter factor means true
physical parameter divided by the reported nominal value. Mixed scenarios
overlay one power-channel fault on the physical/topology base acquisition.
Separate lower-severity sensitivity profiles should not be described as this
completed DAgger population.

## 5. Physical HIF units

Voltage is line-to-line and the power base is three-phase. The same physical
resistance corresponds to very different per-unit values at different voltage
levels. Main training focuses on 69-kV branches. The main DAgger implementation
uses a normalized DSS equivalent with an explicit conversion boundary; the
standalone physical exporter uses the actual local kV. Bus 8 has no eligible
same-voltage midspan line in the selected IEEE-14 topology. IEEE-57 has a
separate declared reconstruction and separate detection studies, not the
DAgger results on these slides.

## 6. Physical corrections

Three fixes address model consistency: shunt conventions, paired reference
dispatch, and generator reactive limits. OpenDSS generator kW writes had
overwritten the limits, so synchronous condensers could not regulate as
intended. This produced a dataset cue correlated with fault family. Restoring
limits changes the actual simulated observations, so old frozen measurements
must remain attached to the old model. The latest 20260923b rebuild also uses
a tighter shared solve tolerance. Strict physics/current checks pass, while
five known legacy NLM top-1 ranking misses remain.

## 7. Admission and detection limits

The 131 admitted windows are selected from 420 main HIF windows. The six raw
HIF cohorts also include detection-limit and sweep data, bringing the raw
total to 777 windows. The selected population is heavily weighted toward
stronger physical signatures: no 500–1000 ohm case is admitted. Three admitted
windows share an alarm with their paired healthy observation, so a WLS alarm
alone is not a causal HIF label. Keep weak/unresolved cases available for
evaluation rather than removing them from performance denominators.

## 8. Completed training run

The model is the configured Gemma 4 12B instruction model. Distinguish physical
roots from visited states and training rows. R2's replay pool includes D0 and
R1 rows; the new mixture uses 660 replay and 660 newly collected labels.
R1 previously used 1,376 rows and 344 optimizer steps. The completed historical
run had initial HIF flags and auxiliary telemetry. D0 did not contain eligible
mixed-HIF or healthy-telemetry training labels; each DAgger round contributed
only six mixed-HIF rows. The later continuation behavior was enabled by the
runtime without retraining the checkpoint.

## 9. Original DAgger results

All policies use the same 160 development roots. R2 fixes a parameter case and
a multiple-meter case but regresses on a mixed meter/parameter case. The small
net difference is not evidence of a statistically robust advantage. A resolved
waveform diagnosis is a task outcome, not physical fault removal. Handoffs can
be truth-correct under the task contract.

## 10. Revised-code reevaluation

The unchanged R2 weights produce 157 audit-correct final states, but four do
not terminate. Report 153 when requiring both correctness and terminal closure.
The expert has 158 correct terminal outcomes. Eight gains simply repair the
healthy-control reference; those original acquisitions were already preserved.
The other eight gains are genuine gross-meter repairs while HIF remains.

## 11. HIF contribution compensation

The user-facing idea of masking is implemented by subtracting an estimated
physical contribution from a temporary WLS input. No sensor channel is
discarded and no residual is set to zero. Equivalently, WLS uses a balanced
forward model shifted by the estimated HIF effect. This is conditional on the
HIF fit and the known operating-point context; it is not joint estimation of
state, HIF and arbitrary meter errors. The current SCADA scan is held out of
the HIF fit to reduce absorption of the current gross meter error.

## 12. Meter continuation and acceptance

Five-sigma conditional meter evidence is a different quantity from the WLS
normalized-residual threshold of four. The prediction range comes from the
best fit and sampled corners of the near-best parameter rectangle. It is not
a confidence interval. Broad or uncertain discrepancies require handoff.
Verification checks physical constraints as well as conditioned WLS. The
exception for existing voltage violations requires unchanged voltage readings;
it does not reclassify the physical state as healthy.

## 13. Verification population

The 24 episodes are variants of eight roots: original mixed cases, extra
overlap stress cases and HIF-only controls. They are not 24 independent faults.
The original mixed expert episodes each take nine actions and end in handoff.
Both before/after J ranges on this slide are conditioned statistics, so the
comparison measures meter repair after HIF accounting. Two overlap stress
cases were detected but correctly rejected because their voltage edits did
not satisfy physical acceptance.

## 14. Same-channel example

The HIF contributes about 0.152163 pu at bus 1's active-power injection, while
the extra meter error is -0.1 pu. The original noisy HIF-present reading is
about 2.098634 pu; after adding the error, the observation is 1.998634 pu.
Subtracting the estimated HIF effect gives 1.846471 pu, so the meter bias
remains visible. The raw repaired target is the HIF-present model value,
2.100483 pu. This paired stress episode contains two faulty meters; both are
repaired, and no healthy reading changes. The physical HIF is retained.

## 15. Remaining policy failures

Four revised R2 episodes reach the right meter state but generate unsupported
handoff arguments. Repeated nonadvancing attempts stop them at 16–18 actions,
before the 40-action limit. Three other roots remain wrong. The original
16 false-finalization counts represent accepted finalizations later failing
audit; the revised 26 represent rejected requests. Avoid treating all these
counters as applied harmful edits or as interchangeable metrics.

## 16. SCADA-only discovery

The new requirement permits only balanced SCADA evidence. The old discovered
mode could still acquire phase or harmonic data after WLS, so a strict evidence
profile was necessary. The initial eight-root check demonstrates alarm and
action invariance under hidden-data changes. It does not show successful
HIF identification or mixed repair. Existing auxiliary HIF compensation is
disabled; a new SCADA-only hypothesis model needs its own validation.

## 17. Next experiments

Freeze the completed physical corpus and model together, then recollect
compatible expert labels. Preserve the current fixed checkpoints as baselines.
Compare policies on matching roots before crediting retraining with a gain.
SCADA-only HIF conditioning must explicitly test competing meter errors,
identifiability, overlap and prediction uncertainty.

## 18. Family table

Every column has the same 160-root denominator. The revised columns reflect a
runtime/audit change, not new learned weights. In revised R2's mixed-HIF row,
eight states are correct but only four terminate. The full correct-terminal
total is therefore 153. Parameter failures differ between the expert and
learned policy despite their similar aggregate score.

## 19. Metric definitions

There are three separate questions: is the task-specific final state correct,
did the policy complete a valid terminal outcome, and did the episode satisfy
the stricter stored completion certificate? Physical removal is another
question again. Use 157, 153 and 136 only with their respective definitions.
The source index identifies the exact saved reports and physical-model scope.
