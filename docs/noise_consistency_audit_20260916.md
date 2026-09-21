# Measurement-noise consistency audit

**Historical audit:** the implementation gaps below have since been addressed
for current generation and execution paths. See `noise_alignment_20260916.md`
for the changes and fresh validation. The findings below retain the earlier
source/artifact context and do not describe the newly aligned generators.

Fresh audit in `C:\Users\Holiday\Documents\ChatGPT\PSSE_Agent` at HEAD
`a6550caa70ce40515785e03241c9307c6317eafb`, 2026-09-16. The prior HIF fits
were generated at `57e6898`; their provenance is retained separately from
the new result labeling and covariance validation.

**Cross-family consistency is not currently guaranteed.** Some paths have
explicitly matching generation and estimation covariance; several historical
and scenario-specific paths do not. This audit does not relabel old data,
change old noise draws, or claim all generators have now been migrated.

**The very small HIF statistic was a noiseless baseline check**

The evaluator previously used the compact key `event_absent` for a deterministic
OpenDSS forward prediction. That array contains no sampled sensor noise. Its
near-zero WLS residual establishes measurement-model compatibility after the
capacitor/injection convention adjustment. It does not demonstrate that
repaired noisy observations have almost no residual.

| Input to WLS | Original corpus median J | Stress corpus median J |
|---|---:|---:|
| Noiseless HIF-absent model prediction | 0.00008605 | 0.00003926 |
| Noisy event-only data with fitted HIF effect subtracted | 91.3712 | 90.8128 |
| Actual repaired transient-error data | 90.5157 | 90.4203 |
| Actual repaired persistent-error data | 90.5157 | 90.4203 |

Original actual post-repair J ranges from 72.7999 to 111.6514. Stress actual
post-repair J ranges from approximately 73.8234 to 108.5720. These are the
stored WLS probes: 68+34 original and 24+12 stress cases, not every exhaustive
meter placement.

Each repair changes one channel and retains the noise in the other 121.
Replacing a corrupted channel with a model prediction removes that channel's
particular sensor-noise realization; this changes the residual distribution.
Compared with the same root's noisy event-only control, the mean change in
transient J is -0.8653 on original data and -0.4266 on stress data. It does
not explain a statistic near 1e-4.

For raw, correctly modeled independent Gaussian observations, WLS uses
J=e^T R^-1 e with R=diag(sigma^2), and approximately E[J]=m-rank(H)=95 here.
The pilot's voltage sigma is 0.001 pu and power sigma is 0.01 pu. Its global
alpha is 0.05, threshold 118.7516118 for 95 nominal degrees of freedom, and
the separate normalized-residual threshold is 5. The anomaly rule is inclusive
global OR local threshold crossing. Fitted-effect uncertainty, selected meter
replacement and post-selection change the exact distribution; matching raw
sensor sigmas does not justify calibrated post-repair chi-square p-values.

**Confirmed generation/weighting differences**

| Path | Applied noise | Estimator convention | Finding |
|---|---|---|---|
| Fresh balanced no-error, meter and parameter scenarios | Vm 0.001 pu, P/Q 0.01 pu independent Gaussian | Same sigma vector, with explicit builder assertion | Consistent declared SCADA covariance |
| New GNN physical corpus/loader | Explicit noiseless mean, sigma vector and noise seed; exactly one draw | Same sigma passed to WLS | Consistent declared SCADA covariance |
| Original HIF SCADA/current channels | At recorded noise_scale=1, SCADA 0.001/0.01 and current real/imag 0.001 | Same sigmas | Consistent on inspected corpus |
| Original HIF phase voltages | No added noise | Multiscan fit assumes 0.005 per real/imag component | Confirmed mismatch |
| New HIF stress phase voltages | 0.005 per real/imag component | Same 0.005 | Consistent |
| Legacy unbalance SCADA and phase voltages | Direct noiseless OpenDSS outputs | SCADA WLS uses positive nominal sensor variance | Confirmed mismatch; tracked branch currents separately have 0.001 component noise |
| Full node-breaker topology aggregated injections | Sum of independently noisy physical meter readings | Balanced WLS uses fixed 0.01 for each operator injection | Missing covariance propagation after aggregation |
| Harmonic complex voltage phasors | Real/imag sigma 1e-4/sqrt(2); complex RMS 1e-4 | HSE uses 1e-4 separately on each real/imag row | Assumed component variance is twice actual |

The topology issue occurs in the actual normal 14-bus operator layout. Bus 3
sums two distinct injection meters (`3B1`, `3B2`), so its P and Q injection
sigmas are sqrt(2)*0.01=0.0141421, while WLS uses 0.01. Its actual variance
is 0.0002 versus assumed 0.0001. Twelve buses have one injection meter each.
Bus 7 is an intentional structural zero and is handled by the solver's
zero-injection treatment; that is distinct from forgetting to noise a sensor.

Different SCADA and phasor sensor accuracies are legitimate. The problems are
undeclared differences between applied noise and the covariance used for the
same channel, not the mere use of different sigma values for different sensors.

Other configuration boundaries:

- The old scenario generator estimates a per-index noise profile from 200
  no-error rows for error floors and repair tolerances, while runtime WLS
  retains fixed nominal sigmas. Fresh balanced generation uses the declared
  registry vector instead.
- HIF `noise_scale` and topology noise scaling can change applied noise
  without changing the generic WLS defaults. A legacy coupled
  parameter/topology helper also contains half-noise generation; this audit
  did not establish that it generated the latest experiments.
- Recoverable-suite admission can require noisy clean controls to pass WLS
  and corrupted controls to fail it. Such retained cases are selected, not
  an unfiltered population for calibrating false-alarm rates.
- The HIF pilot uses a 5-sigma local threshold. Other current entry points
  allow different thresholds or chi-square-only detection. A unified run
  must record resolved detector settings rather than inherit defaults.
- Unbalance `z_true` and original HIF top-level `z_true` are balanced reference
  models, not the same-physics noiseless observation. Subtracting them from
  disturbed observations would confound physical differences with sensor noise.

Primary source locations: `psse_env/systems/registry.py::measurement_sigma`,
`psse_env/providers/balanced_corpus.py::build_balanced_corpus`,
`research/gnn_screen/dataset.py`,
`Transmission/generate_measurements_hif_ieee14.py` (scan construction),
`Transmission/generate_measurements_imbalance.py` (direct observation export),
`Transmission/ieee14_full_substation.py::add_telemetry_noise` and
`operator_vector_for_layout`, `Transmission/generate_hse_traces.py` (phasor
noise), `Harmonics/hse_utils.py::estimate_single_source_injection_from_voltage`,
and `tools/lagrangian_port.py` (variance construction and objective).

**Empirical and fresh-control checks**

For 20,740 original HIF SCADA noise components, the standard deviation after
division by declared sigma is 1.00047. For 40,800 original current components
it is 1.00484. Stress SCADA/current/voltage component values are 1.01378,
1.00050 and 1.00131 respectively. These support the declared sensor covariance
for those channels; they do not validate the original noiseless phase voltages.

A fresh control uses a noiseless HIF-absent OpenDSS mean with the matching
branch-net measurement operator. Four hundred independent Gaussian draws
are generated without WLS admission filtering, seed 20260916. The same noise
draw is paired across the three diagnostic conditions:

| Control condition | Mean J | Global alarms out of 400 | Local alarms out of 400 |
|---|---:|---:|---:|
| Applied sigma matches WLS sigma | 95.7117 | 28 | 0 |
| Applied sigma is half the WLS sigma | 23.9280 | 0 | 0 |
| Matched noise, then one channel replaced by its model mean | 95.2312 | 25 | 0 |

The first mean agrees with the expected scale of approximately 95; the second
agrees with 95/4=23.75. The noiseless baseline gives J=0.00004507. The observed
global false-alarm fraction is 7.0% at nominal 5%, not proof of exact 5%
calibration. This finite, single-operating-point control is a scale check;
population calibration needs more operating points and the actual fitted
correction pipeline. No thresholds were tuned to these results.

**What is now enforced, and what remains necessary**

Added `psse_env/noise_contract.py` with two validators:

1. `validate_noise_channel` distinguishes noisy observations, noiseless
   references and model predictions; normalizes explicit component versus
   complex-RMS sigma; compares applied and estimator component covariance;
   and rejects unknown/mismatched/non-Gaussian declarations in strict mode.
2. `validate_shared_scada_covariance` rejects a scan sigma override that differs
   from the root sigma used by the HIF meter detector/WLS, as well as invalid
   shapes, null overrides and nonfinite data. It is wired into fitting and
   evaluation, including cached-fit reanalysis.

Receipts explicitly say that declaration consistency does not certify actual
source sampling or population calibration. The source-inspection cases in
the audit are checked separately from empirical noise and Monte Carlo data.

The HIF evaluator now emits `input_role` and `statistical_interpretation`
with every WLS result. Noiseless baseline checks cannot be confused with
noisy post-repair statistics in these refreshed artifacts. Reanalysis retains
the original fit-generation commit and records the current analysis commit;
source data and numerical settings must still agree.

A defensible common future experiment still requires migration of the affected
generators/consumers: apply and declare noise on all intended sensors; propagate
R through any aggregation as A R A^T; pass the resulting covariance into the
estimator; resolve complex-component conventions explicitly; retain unaffected
noise through repairs; and calibrate raw and post-repair controls separately.
Older corpora with different noise contracts should be regenerated or reported
as separate cohorts. No universal historical consistency guarantee is claimed.

Reproduce the audit and tests:

```powershell
python scripts/audit_measurement_noise_consistency.py --draws 400
python -m pytest -q -p no:faulthandler tests/test_noise_contract.py tests/test_hif_recovery_experiment.py tests/test_conditioned_meter_recovery.py test_hif_conditioned_recovery.py
```

The machine-readable audit is `output/noise_consistency_20260916/audit.json`.
All 61 tests and 7 subtests passed at the final integrated check, including
32 new declaration/covariance validation cases. The existing HIF recovery
results were re-evaluated with saved fits, explicit statistic roles and the
shared-SCADA check; their numerical recovery results did not change.
