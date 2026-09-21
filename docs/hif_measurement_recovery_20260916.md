# HIF plus measurement-error recovery experiment

**Historical noise configuration:** these recovery counts precede the common
noise/covariance alignment in `noise_alignment_20260916.md`. They remain an
audit of the original inputs, not a performance claim for regenerated aligned
datasets. Fresh evaluator runs now prepare input noise explicitly and require
a new output directory when the preparation contract differs.

Fresh local experiment, 2026-09-16. Repository:
`C:\Users\Holiday\Documents\ChatGPT\PSSE_Agent`, branch
`codex/wls-screen-gnn`, starting HEAD `57e6898b7c57d20cab96d142683eb1e1671b694e`.
All work was performed in this repository, not `C:\dw`.

**Statistic clarification from the subsequent noise audit:** the value
J=0.00004507 is a *noiseless forward-model compatibility check*. It is not
the statistic of repaired noisy measurements. Actual sampled post-repair
J has median 90.52 (original corpus) and 90.42 (stress corpus), with ranges
72.80–111.65 and 73.82–108.57 respectively. The estimator retains the noise
on all 121 unmodified readings. See `noise_consistency_audit_20260916.md`
for the cross-family inconsistencies, fresh Gaussian controls, and the
limits of the current experimental configuration.

**Result:** an estimated HIF-present forward model can support recovery of a
coexisting SCADA meter error, including errors on channels strongly affected by
the HIF. This is demonstrated on synthetic data with known operating points
and a matching OpenDSS model. Physical-effect prediction, fault localization,
meter repair, and actual physical fault removal are separate outcomes. The
HIF is retained in the physical model throughout this experiment.

The new code is a reusable physical replay helper, a conditional meter-repair
helper, and a reproducible experimental evaluator. It does not enable a new
agent action, relax the release audit, or change existing provider/expert
finalization. The present experiment establishes a basis for that subsequent
integration; it is not a new DAgger or student-rollout success result.

**Method and evidence boundaries**

For each event, observable terminal-current evidence ranks the candidate
branch; the existing multiscan HIF estimator fits branch phase, along-line
position, and shared resistance using scans 1–4. Scan 0 is excluded from all
fits and used for heldout evaluation. The fit uses a 7-by-9 coarse grid, two
refinement starts and at most 25 local function evaluations. Current operating
points are supplied as trusted observable context, not estimated from the
corrupted meter. The runtime scan input is an explicit whitelist excluding
clean arrays, clean currents, labels and stored truth-derived diagnostics.

The replay helper solves HIF-present and HIF-absent conditions at the same
operating point and preserves all 122 external channels. The vector contains
phase-A voltage magnitudes and three-phase summed powers. A hidden HIF split
node does not change the original external meter identity.

The conditional meter test compares active observations with the predicted
HIF-present readings. It proposes only sparse corrections outside a sampled
prediction envelope, and replaces a bad meter with the HIF-present prediction.
Healthy readings are retained exactly. The sampled envelope contains the best
fit and the corners of the near-best alpha/resistance profile rectangle; it
is neither a guaranteed nonlinear bound nor a calibrated confidence interval.

Thresholds were declared before the full runs:

- Meter detection: discrepancy beyond the sampled envelope greater than
  5 measurement sigmas. These are measurement-standardized discrepancies,
  not normalized WLS residuals.
- Candidate prediction-envelope full width: at most 2 measurement sigmas.
- Successful meter audit: exactly the injected channel changes, no healthy
  channel changes, and repaired value within 3 sigmas of the noiseless
  HIF-present reading. Error to the original noisy reading is also recorded.
- HIF audit: correct branch and phase, alpha error at most 0.15, resistance
  relative error at most 20%. No tolerances were widened after results.
- Material overlap: HIF measurement contribution at least 1 sensor sigma.
  Thus "nonoverlap" is a below-resolution approximation, not literal zero
  physical interaction.

The transient experiment injects both signs of a 10-sigma error independently
at every one of the 122 heldout SCADA channels. For persistent-error stress,
the same positive 10-sigma bias is also injected into every fitting history
scan; the estimator is rerun. Two persistent targets per event are chosen
from the maximum and minimum estimated HIF-effect channels. Auxiliary phase
voltage/current sensors are not corrupted by these SCADA-error injections.

All attempted roots, unsuccessful fits and unexecuted traces remain visible
in the results. Repeated meter placements within one event are dependent
trials; 4,148 placements are not 4,148 independent physical events.

**Fresh results**

| Experiment | Physical events | Meter-error trials | Meter recovery | Correct HIF diagnosis plus meter recovery | Healthy-channel writes |
|---|---:|---:|---:|---:|---:|
| Existing current-bearing corpus, transient | 17 | 4,148 | 4,148 | 3,904 | 0 |
| Same corpus, persistent | 17 reused | 34 | 34 | 32 | 0 |
| Stronger HIF stress corpus, transient | 6 | 1,464 | 1,464 | 1,464 | 0 |
| Same stress corpus, persistent | 6 reused | 12 | 12 | 12 | 0 |

There were no missing or unexecuted trials. On diagnostic-only heldout
controls, the conditional meter test proposed no repairs on all 17 original
and all 6 stress roots. This small observed control set is not a population
false-alarm calibration.

The original 17 roots have weak SCADA HIF effects: maxima range from 0.13 to
1.55 sensor sigmas. The physical-model predictor's worst heldout noiseless
measurement error was 0.0448 sigma. Only 18 transient placements across three
roots meet the material-overlap definition. Both the physical method and an
unconditioned HIF-absent predictor pass the 3-sigma meter-accuracy criterion
on all original transient placements. These original cases alone would be
insufficient evidence that conditioning resolves substantial overlap.

The stronger corpus therefore reruns the real OpenDSS simulator for six
distinct branches spanning phases A/B/C with resistance 5 or 10 pu, preserving
the source operating points. Its strongest HIF effect is 29.63 sigmas. It
adds independent Gaussian SCADA, branch-current and phase-voltage noise;
phase-voltage real/imaginary sigma is 0.005 pu. The original corpus's phase
voltages were noise-free, so this is explicitly a stronger auxiliary-noise
stress as well. Generation is deterministic and was reproduced byte for byte.

| Stronger transient stress comparison | All 1,464 placements | 248 materially overlapping placements | Aggregate off-target writes |
|---|---:|---:|---:|
| HIF-present physical prediction | 1,464 recovered | 248 recovered | 0 |
| Nonoverlap fallback with overlap failures retained | 1,216 recovered | 0 recovered | 0 |
| HIF-absent prediction without compensation | 724 recovered | 50 recovered | 968 |

The off-target-write total counts writes across repeated trials, not distinct
physical meters. The absent-model ablation shows why physical conditioning
matters when the HIF effect is substantial. The fallback also rejects all
six materially overlapping persistent stress cases, while the physical
method recovers all 12 persistent cases. The worst heldout noiseless
prediction error in the stronger corpus is 0.0525 sensor sigmas.

**An unresolved physical-localization case remains a failure**

Original event `ieee14_hif_000015` identifies branch row 15 (line 9–10), phase
C and resistance correctly, but alpha error is 0.3367, exceeding 0.15. Its
external measurement prediction is nevertheless accurate to 0.0117 sigma.
Its 244 transient and two persistent trials remain joint-diagnosis failures.

A separately labeled, deliberately selected followup used all nine history
scans, an 11-by-13 grid and three refinement starts. Alpha error improved to
0.18495 but still failed; the estimator reports `parameter_identifiable=False`
and a near-best alpha interval of [0.05, 0.95]. Heldout prediction error fell
to 0.00475 sigma. This followup is not substituted into the original aggregate.
It demonstrates that an accurately predicted external effect need not imply
uniquely recovered along-line fault position.

**WLS measurement convention and continuation checks**

An independent baseline check found a capacitor/injection convention mismatch.
The OpenDSS exporter includes capacitor supply in `Qinj`, whereas canonical
MATPOWER WLS represents the bus-9 capacitor in `Ybus` and expects the
corresponding generator-minus-load injection. On one noiseless HIF-absent
baseline this produced J=199.06185 and maximum normalized residual 14.11076.

The evaluator now uses the matching branch-net-injection measurement operator:
bus shunts are omitted only from the local WLS observation model, while the
physical OpenDSS circuit retains every capacitor. Observations are unchanged.
This avoids propagating a potentially faulty Vm reading into a transformed
Qinj reading. The same baseline gives J=0.00004507 and maximum normalized
residual 0.003159 after aligning conventions. The unadapted statistics are
retained in each result. Existing general-purpose WLS/provider behavior is
unchanged.

The global statistic is e^T R^-1 e, using the corpus sensor variances. With
122 measurements and 27 states, dof=95, alpha=0.05 gives threshold
118.75161. The separate normalized-residual threshold is 5. An alarm occurs
when either statistic reaches its threshold. After HIF compensation, all
23 diagnostic-only controls are quiet under both tests; original J values
range from 73.20 to 111.64 and stress J values from 74.14 to 109.55.

WLS continuation is probed on maximum/minimum HIF-effect meter locations:

| Experiment | WLS probes | Alarm before meter repair to quiet after | Quiet after repair |
|---|---:|---:|---:|
| Original transient | 68 | 66 | 68 |
| Original persistent | 34 | 34 | 34 |
| Stronger transient | 24 | 24 | 24 |
| Stronger persistent | 12 | 12 | 12 |

Two original transient cases were already quiet under the WLS detector before
repair; the independent physical-model discrepancy test still found the bad
meter. Intersecting alarm-to-quiet transitions with correct full HIF diagnosis
gives 62/68, 32/34, 24/24 and 12/12 respectively. WLS was not rerun for every
one of the exhaustive placements. Compensation-estimation uncertainty is not
included in these WLS covariance calculations; these are diagnostic probes,
not a claimed recalibration of the nominal chi-square false-alarm rate.

**Implementation and reproducibility**

- `three_phase_nlm/hif_conditioned_recovery.py`: paired physical replay,
  snapshot/channel binding, strict per-scan resistance handling.
- `three_phase_nlm/conditioned_meter_recovery.py`: HIF-present meter proposals
  and a conservative nonoverlap fallback; preserves input observations.
- `scripts/evaluate_hif_measurement_recovery.py`: whitelisted fitting,
  transient/persistent tests, independent audits, WLS diagnostics, per-root
  artifacts and resume support.
- `scripts/build_hif_recovery_stress.py`: reproducible physical stress corpus.
- `scripts/plot_hif_measurement_recovery.py`: same-channel residual figure.

From the repository root:

```powershell
python scripts/evaluate_hif_measurement_recovery.py --output-dir output/hif_measurement_recovery_20260916/development_17 --persistent-targets 2
python scripts/build_hif_recovery_stress.py --output-dir output/hif_measurement_recovery_20260916/stress_corpus
python scripts/evaluate_hif_measurement_recovery.py --samples output/hif_measurement_recovery_20260916/stress_corpus/samples.jsonl --output-dir output/hif_measurement_recovery_20260916/stress_6 --persistent-targets 2
python scripts/plot_hif_measurement_recovery.py --experiment output/hif_measurement_recovery_20260916/stress_6
python -m pytest -q -p no:faulthandler test_hif_conditioned_recovery.py tests/test_conditioned_meter_recovery.py tests/test_hif_recovery_experiment.py
```

`--reuse-fits` recomputes decisions and WLS probes from saved fits/replays.
Completed roots otherwise resume without recomputation. Tests passed:
**27 tests and 7 subtests**, including real OpenDSS replay, same-channel error
preservation, overlap rejection, healthy-channel preservation, persistent
injection identity, missing-result denominators and the capacitor convention.

Evidence is under `output/hif_measurement_recovery_20260916/`: the two run
directories each contain `summary.json`, `root_results.json`, per-root fits,
replays and `traces.jsonl`. The stress corpus has generation and validation
receipts; `localization_followup_root15.json` contains the separate unresolved
localization experiment. `stress_6/same_channel_recovery.png` and `.svg` show
the largest-effect channel with an additional meter error.

The existing mixed-scenario generator's separate HIF metadata can retain a
pre-injection SCADA copy. This experiment avoids that ambiguity through
explicit heldout/current identity and, for persistent tests, corruption of
every fitting scan. Integrating the method into the agent still requires
current-state conditioning, refreshed evidence after corrections, a mixed
diagnostic-and-repair audit, and closure checks that do not equate accurate
measurement prediction with unique HIF localization or physical removal.

The first physical recovery attempt supports using the HIF-present model as
the primary continuation method. The independence fallback remains available
with explicit overlap failures. Uncertain operating points, corrupted
auxiliary phasors, model mismatch, multiple simultaneous meter errors and
other physical fault families remain outside these results.

## Physical resistance update (2026-09-20)

Historical HIF pu values and unmarked `r_hif_ohm` labels in this document describe the normalized 1 kV model. New physical-ohm corpora use local voltage bases and explicit measurement conventions. See [the reconfiguration and fresh results](ieee14_hif_legacy_reconfiguration_20260919.md); old corpora and historical measurements remain unchanged.
