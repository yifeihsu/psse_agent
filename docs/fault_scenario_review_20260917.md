# Reviewed fault scenario profile

The [physical IEEE14 HIF profile](ieee14_physical_hif_20260918.md) now supplies
the bundle default, with explicit 69/13.8/18 kV bus bases and faults specified
in ohms. The `reviewed_v1` settings below remain available explicitly for
reproduction; their historical artifacts retain their original definitions.

The review is implemented as the versioned `reviewed_v1` research profile.
It creates new physical scenarios and preserves the existing SFT/DAgger source
files, practical GNN benchmark, trained checkpoints, and published results.
The shared numerical settings are in `psse_env/fault_profiles.py`.

The subsequent [WLS-observable training admission](wls_observable_training_20260917.md)
now creates the default training views. Raw physical manifests remain available
for evaluation and audits; revised training requires the filtered observed
windows with actual WLS evidence and executed expert prefixes.

| Family | Revised treatment |
| --- | --- |
| Resistive HIF surrogate | Main bands 5–10, 10–20, and 20–40 pu, uniform within each band. Early curriculum gives the first two bands twice the representation of the third; the full curriculum gives equal representation. No resistance below 5 pu. Location remains 0.25–0.75 from the from-end. |
| Weak HIF | A separate 20–200 pu evaluation population on validation/test parents. It never enters the main training manifest. |
| Parameter | Existing gross physical/reported factors 0.1–0.5 or 2–5 remain unchanged. Separately labeled moderate sensitivity factors are 0.70–0.95 or 1.05–1.30, for R, X, and RX. |
| Additive measurement error | Existing strong gross-error benchmarks remain. Each stored scenario records its actual channel indices, units and offsets; an alarm in a mixed case is not attributed automatically to its physical component. |
| Measurement chain sensitivity | Separate coherent complex-power transforms model CT/PT gain and angle errors. The companion currently uses CT gains 0.95/1.05 and CT angles −1/+1 degree, explicitly declared research assumptions, not instrument specifications. These cases do not receive the additive 10-sigma floor. |
| Unbalance | Preserve total P/Q and symmetric Dirichlet(3,3,3) phase draws. Target achieved maximum VUF strata below 1%, 1–2%, 2–3%, and above 3%. Low cases and severe stress cases have separate manifests; valid off-target candidates remain positive. All phase orientations remain eligible. |
| Harmonic | Keep 10–20% source-bus voltage THD as stress and add 1–5% sensitivity cases. Export and test both fundamental-only and true-RMS voltage metering. Existing active/reactive power-transducer semantics remain explicit. |
| Topology | Preserve the binary status error and actual circuit re-solve. Record electrical impact; do not multiply a binary status or stored residual. Keep the branch-status GNN adapter distinct from the full node/breaker companion. |
| Noise | Baseline power sigma 0.01 pu; separate 0.005 and 0.002 pu accuracy scenarios. Voltage sigma remains 0.001 pu. Covariance and actual Gaussian draws change together. No field-accuracy claim is made. |

The parameter direction is actual physical value divided by the reported value.
Moderate factors represent a declared sensitivity experiment; they are not
presented as a field-measured probability distribution or a specific
temperature/line-length model.

## Generate a new bundle

From the repository root:

```powershell
python -m research.reviewed_fault_scenarios --output-dir output/reviewed_fault_scenarios_new --curriculum-stage early --train-parents 1 --validation-parents 1 --calibration-parents 1 --test-parents 1
```

Use a new output directory for every run. The defaults are a small generation
and integration check, not a statistically powered benchmark. Increase the
parent counts for a larger experiment, and use `--curriculum-stage full` for
equal main HIF-band representation. The attempt cap bounds physical/VUF
proposal retries; the reports retain any coverage shortfalls instead of
claiming every requested stratum was filled.

The bundle contains:

- `baseline/core/manifest.jsonl`: the selected five-family main population.
- Separate core boundary, weak-HIF, parameter-sensitivity and unbalance
  sensitivity/stress manifests, plus the full proposal ledger.
- `harmonic_scenarios.jsonl`: same-parent healthy, low-THD and stress snapshots
  under both voltage-meter definitions and all three noise profiles.
- `measurement_chain_manifest.jsonl`: separately labeled P/Q-coupled CT
  sensitivity examples, usable by the existing five-family manifest loader.
- `paired_attribution_manifest.jsonl`: healthy, physical-only, meter-only and
  mixed arms with the same within-parent standardized Gaussian noise. Their
  distinct window IDs remain intact; an explicit noise-group identity controls
  the pairing in the actual dataset loader.
- `energy_balanced_training_manifest.jsonl`: an optional equal-count sample
  across the available offline energy bins. Its report names missing bins and
  retained families. It is not automatically used for training and does not
  replace the complete main population or the separately retained weak cases.
- `node_breaker_scenarios.jsonl`: full physical breaker/status cases and controls
  with fixed physical-meter identity and propagated covariance, including exact
  structural-zero rows. This adapter is separate from the GNN branch-status one.
  Observed raw substation telemetry is retained for subsequent model re-rendering;
  it is never reconstructed from the hidden clean source. The finite-impedance
  breaker solve can have a small nonzero healthy residual against an ideal
  contracted operator model, which is recorded rather than forced to zero.
- `accuracy_views/`: the same core physical means, fault offsets, parents,
  splits and noise seeds, with alternative sigma vectors. Cohort membership is
  frozen from baseline; it is not reselected to make the accuracy comparison
  easier. The dataset loader applies noise once and passes the same covariance
  to WLS and graph construction.
- `fault_profile.json` and `bundle_report.json`: configuration and provenance.

The underlying GNN generator also accepts `--scenario-profile reviewed_v1`,
`--stage early|full`, and `--noise-profile baseline|accuracy_005|accuracy_002`.
Its legacy profile remains available for reproduction. A newly generated
standalone accuracy-profile cohort can have different admissions; use the
bundle's frozen accuracy views for a paired sensing comparison.

The accuracy profiles change SCADA injection and terminal-flow noise. Acquired
phase-voltage/current assumptions, harmonic phasor noise, and auxiliary
breaker-flow noise retain their baseline settings. In particular, the full
node/breaker companion keeps breaker-flow sigma at 0.01 pu while propagating
the selected injection/terminal-flow covariance into the operator vector.

## Physical severity and available signal are separate

Offline audits record the physical descriptors and the nonlinear WLS objective
on exact faulted means using the reported model and declared covariance:

`J_exact = (z_exact - h(x_hat)).T @ R^-1 @ (z_exact - h(x_hat))`.

The descriptive energy bins are `[0,1)`, `[1,9)`, `[9,25]`, and `(25,infinity)`.
The nonlinear solver provides a converged local fit, not a proof of the global
minimum. A failed fit is `unavailable` with its error, never zero or healthy.
These values, VUF, simulator state and physical truth remain outside online GNN
features. The main cohort retains its explicit paired-mean visibility rule;
`J_exact` is not an extra admission threshold.

HIF records distinguish resistance, current, voltage, dissipation and local
line-impedance ratio. Parameter records distinguish R/X/RX, absolute changes
and branch loading. Unbalance records distinguish requested fractions,
actual source phase powers, source-bus VUF and network-maximum VUF. Fault family,
physical severity and WLS-visible signal should be reported separately.

The 1 kV/100 MVA realization has `Zbase = 0.01 ohm`. Its resistive fault power
uses `P_fault_pu = |v_phase_pu|^2 / (3*R_pu)` with phase-to-neutral voltage base.
Changing voltage bases changes dimensional ohms and amperes; no 138 kV branch
assignment is assumed. These are steady-state resistive surrogates, not arcing
fault simulations. Some commercial HIF detectors explicitly use arcing
signatures, as described by [SEL](https://selinc.com/products/751/).

The harmonic companion uses the existing frequency-domain harmonic network
model initialized from each solved fundamental parent. Fundamental/RMS voltage
selection does not change its existing power transducer into a fundamental-only
power meter. Reactive power uses the declared legacy quadrature response.
Separately attached harmonic contexts are not claimed to be fully coupled
simulations. [EPRI's harmonic-flow documentation](https://opendss.epri.com/HarmonicFlowAnalysis.html)
also distinguishes the fundamental initialization and harmonic network/source
solution. The review's planning/THD standards are not used as universal fault
thresholds or certification criteria.

## Comparison discipline

Changing the test population and improving a trained detector are different
experiments. First compare a frozen model and frozen threshold across named
populations. To claim a training improvement, compare old and newly trained
models on identical frozen evaluation manifests, including weak HIFs and
moderate/low-severity cases. Calibration parents remain healthy-only and are
disjoint from training, validation and test parents. These changes alone do
not retrain a model or establish improved recall.

## Executed validation

The fresh bundle is at
`output/reviewed_fault_scenarios_20260917/validated_bundle/`. Its four-parent
core pilot generated 58 main rows, 46 boundary rows, two weak-HIF evaluation
rows, six moderate-parameter rows, 22 low-VUF rows, and three severe-VUF rows.
Companions contain 60 harmonic meter/noise-profile rows, 30 full node/breaker
rows, 12 CT measurement-chain rows, and 36 matched-attribution rows.

The actual dataset noise/WLS/graph loader accepted 956 noisy windows across the
baseline, accuracy views and compatible companions, with zero invalid windows.
These include related variants and repeated noise draws; they are not 956
independent physical events or completed agent episodes. Saved observed
substation telemetry reproduces its operator measurements exactly. All
recorded source hashes matched the executed files during the audit.

The focused regression suite passed 173 tests and two subtests. The subsequently
added path-backed energy-view regression passed separately, for 174 distinct
tests in total. `git diff --check` passed. The Windows combined pytest process
emitted native DSS import-order diagnostics but completed with zero test
failures; the standalone generation and audit initialize DSS before Torch.

Coverage is explicitly limited: the main slots were filled, but one requested
above-3% VUF test-parent slot reached its 24-attempt cap, and one nonconvergent
physical proposal was logged. Severe cases found during other proposals remain
in the stress manifest. The optional energy-balanced training view has no
fault example below J=1 and does not cover every family in this small pilot.
It is a separate sampling demonstration, not the default training population.

See `bundle_report.json`, `execution_validation.json`, and the parent proposal
ledger for the numerical records. A larger parent population and actual frozen
model comparisons remain necessary for detection-performance claims.

## Physical resistance update (2026-09-20)

Historical HIF pu values and unmarked `r_hif_ohm` labels in this document describe the normalized 1 kV model. New physical-ohm corpora use local voltage bases and explicit measurement conventions. See [the reconfiguration and fresh results](ieee14_hif_legacy_reconfiguration_20260919.md); old corpora and historical measurements remain unchanged.
