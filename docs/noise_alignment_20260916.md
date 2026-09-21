# Applied-noise and covariance alignment

**Detection follow-up:** `waveform_detection_audit_20260916.md` records fresh
fault/control tests. Raw sensor-noise matching does not by itself establish
reliable detection: the current unbalance significance floor omits voltage
uncertainty, and generic WLS still needs the OpenDSS injection-convention
adapter. These issues were exposed by the subsequent detector audit.

Implemented in `C:\Users\Holiday\Documents\ChatGPT\PSSE_Agent`, branch
`codex/wls-screen-gnn`, starting HEAD `a6550caa70ce40515785e03241c9307c6317eafb`.
This supersedes the implementation gaps recorded in
`noise_consistency_audit_20260916.md`. Historical source files and experimental
results remain unchanged; fresh generation and supported source preparation
now publish and consume the covariance of the noise actually applied.

**Supported generation paths covered**

- Classical clean, measurement, parameter, topology and harmonic sources.
- All six existing two-family mixed SFT source combinations.
- HIF, measurement+HIF, unbalance and telemetry-only controls in Round0.
- All 12 current IEEE14 Round0 families, plus the balanced IEEE57 path.
- The new HIF recovery stress generator and its strict SFT source boundary.
- Existing GNN physical-source generation and single-noise dataset loading
  were reviewed and retained; they already pass the same declared covariance
  into WLS. Logical-topology covariance sampling/projection was also already
  consistent and retained.

**Sensor conventions**

| Measurement | Standard deviation at unit scale | Meaning |
|---|---:|---|
| SCADA voltage magnitude | 0.001 pu | Per scalar observation |
| SCADA P/Q injection and branch flow | 0.01 pu | Per scalar sensor, before any physical-meter aggregation |
| Acquired three-phase voltage | 0.005 pu | Each independent real/imaginary component |
| Acquired branch current | 0.001 pu | Each independent real/imaginary component |
| Harmonic voltage phasor | 0.0001 / sqrt(2) pu | Each real/imaginary component; complex RMS remains 0.0001 pu |

For HIF/unbalance, a positive `noise_scale` multiplies both applied noise and
published sigmas. At scale 1.7 the four corresponding sigmas are 0.0017,
0.017, 0.0085 and 0.0017. Invalid, zero or negative noisy-generation scales
are rejected. Deterministic forward predictions remain available separately
and are not relabeled as Gaussian sensor observations.

Each freshly generated source declares its full `sigma_z`. Parameter scan
vectors and alternate topology verification vectors have their own matching
declarations when their dimensions differ. Materialized verification snapshots
carry covariance; a snapshot that preserves the observations preserves their
current covariance too. These fields describe ordinary sensor noise before
the explicitly injected fault or gross error.

HIF and unbalance now add the missing voltage noise. Unbalance also adds the
previously missing SCADA noise. Clean arrays are retained only for offline
audits and are excluded from runtime waveform context. The old half-noise
parameter+topology construction now uses full declared noise. Harmonic
generation retains its original complex random draws, but exports the
component sigma expected by HSE. Its branch-power calculation also now includes
the fundamental; a zero-THD control matches the independent fundamental
measurement function to 7.55e-15.

**Topology covariance is propagated rather than flattened**

Operator observations are read from fixed physical meter identities. When
meters are combined, covariance is propagated as `A R A.T`, including shared
source correlations. In the normal 14-bus layout, bus 3 sums two injection
meters, so its P/Q sigma is `sqrt(2)*0.01`, not 0.01.

Bus-7 P/Q rows [20, 34] have no injection sensor and are exact structural
constraints. They enter the constrained WLS equations and are removed from
the stochastic covariance. No arbitrary small positive variance is substituted.
The normal operator problem has 120 stochastic measurements, 25 effective
state variables and 95 degrees of freedom. Residual covariance is computed in
the constraint tangent space; exact residuals must satisfy 1e-9 tolerance.

Split/merge candidates update the observation layout, covariance and exact rows
together, retaining the original physical telemetry. Aggregate discrepancies
survive a correction when their physical source combination is unchanged. An
unresolvable split of an aggregate meter error produces an explicit ambiguity
failure. Unsupported correlated covariance is rejected by the diagonal balanced
adapter instead of silently dropping off-diagonal entries; the supplied normal,
split and merge layouts use valid diagonal stochastic projections.

The grouped measurement corrector also uses the declared variances and the
constrained residual covariance. It cannot "correct" an exact structural row.
Parameter scans, single/multiscan HIF estimation and harmonic screening/HSE
consume their explicit sensor weights.

**Covariance survives the complete execution path**

The scenario store retains observation covariance. WLS, correction and HIF
public tool wrappers, local trace dispatch and shared argument hydration pass
it to the numerical solver. Covariance/sigma serialization keeps full precision;
small variances are not rounded to zero. The release-envelope schema admits
these observational fields while still rejecting clean arrays and truth labels.

The legacy trace builder no longer replaces noisy parameter observations with
`z_true` during verification. It also rejects a noiseless topology solver
prediction as a substitute for observed verification data. Original observations
can be retained across a model-only change only when explicit unique physical
channel identities match, not merely when array lengths agree.

Known historical HIF/unbalance sources can be prepared using their original
`meta.json` noise evidence. Preparation adds only the missing noise and preserves
existing SCADA/current draws as appropriate. Already aligned sources do not
receive another draw. Missing or ambiguous applied-noise evidence fails with a
regeneration/provenance message. Ambiguous old single-snapshot fallbacks are no
longer silently admitted as matched Gaussian data.

Harmonic source loading requires explicit component/RMS semantics, including
mixed sources. Explicit complex-RMS declarations are converted exactly once;
fresh component sigmas are unchanged. Unspecified old harmonic inputs must be
regenerated rather than guessed from their numeric sigma.

The standalone HIF recovery evaluator also prepares its sources under the
aligned contract before creating new error trials. Preparation version/seed
and source metadata are part of the experimental identity, preventing cached
fits from the earlier noiseless-phase experiment from being silently reused.
The earlier recovery results remain historical and need a fresh aligned run
before being compared with new results.

**Fresh validation evidence**

Artifacts are in `output/noise_alignment_20260916/`.

| Check | Fresh result |
|---|---|
| Classical/mixed source generation | 12 classical rows and all six mixed combinations; explicit covariance on roots and materialized verification snapshots, including changed dimensions |
| Harmonic noise probe | 40 draws; standardized SD 0.99073 for 4,880 SCADA values and 1.00114 for 6,720 phasor components |
| HIF scale-1.7 ensemble | Standardized SD: Vm 1.0389, P/Q 0.9934, voltage components 1.0026, current components 1.0014 |
| Unbalance scale-1.7 ensemble | Standardized SD: Vm 0.9870, P/Q 0.9772, voltage components 0.9758, current components 1.0087 |
| Physical-meter topology projection | 10,000 independent draws; all 120 stochastic sigma ratios 0.98637–1.02008; exact rows zero throughout |
| Round0 construction | All 12 families from both known legacy-current inputs and fresh scale-1.7 inputs |
| Release serialization | 25 envelopes checked: both 12-family sets and a fresh IEEE57 case retaining measurement 490 and all 491 covariance entries |
| Fresh stronger HIF SFT input | Six roots admitted with explicit generated-noise contracts; no reading changed or second noise added at admission |

The unit-scale waveform cohorts are small generation smoke tests. The larger
non-unit cohorts above test the applied/declared scaling. These are covariance
and observation-preservation checks, not claims of population diagnosis accuracy
or nominal false-alarm calibration.

Fresh end-to-end topology checks consumed the declared covariance both before
and after correction:

| Case | Measurement count before -> after | dof before -> after | J before -> after |
|---|---:|---:|---:|
| Dangling line terminal | 122 -> 122 | 95 -> 95 | 8076.981 -> 95.589 |
| Bus split | 122 -> 125 | 95 -> 96 | 3146.506 -> 96.454 |
| Bus merge | 122 -> 119 | 95 -> 94 | 186.026 -> 98.443 |

The main affected-code regression run passed **461 tests and 149 subtests**;
two intentional skips exclude irrelevant dangling/merge parameterizations of
a split-specific aggregate-preservation test. The final tool-forwarding,
trace-protocol, HIF-limit and exact-correction checks passed **82 tests and
16 subtests** after the last legacy fallback changes. Separate final
source-preparation/runtime/release checks passed 57 tests. The standalone
aligned HIF canary completed 244 transient trials on one root, with noisy
phase inputs and unchanged existing SCADA/current draws; this is a boundary
check, not a replacement for the historical recovery benchmark.

Inspect these receipts:

- `classical_harmonic/validation_receipt.json`
- `hif_unbalance/empirical_noise_audit.json`
- `hif_unbalance/round0_canary_receipt.json`
- `topology/covariance_audit.json`
- `topology/provider_canary.json`
- `release_envelope_audit.json`
- `hif_stress_contract/source_admission_receipt.json`
- `hif_recovery_aligned_smoke/summary.json` and per-root noise-preparation receipts

Examples for fresh waveform generation (use new output directories):

```powershell
python Transmission/generate_measurements_hif_ieee14.py --out output/new_hif_aligned --n-hif 4 --n-no-error 2 --scans-per-window 3 --noise-scale 1.7 --seed 20260916
python Transmission/generate_measurements_imbalance.py --out output/new_unbalance_aligned --n-imbalance 8 --n-no-error 2 --noise-scale 1.7 --seed 20260916
python scripts/build_hif_recovery_stress.py --output-dir output/new_hif_stress_aligned
```

Existing generators retain their ordinary entry points. Error severity, diagnostic
thresholds and training/checkpoint results are not retuned by this change.
Matching applied sensor covariance does not itself calibrate post-repair
chi-square p-values: fitted replacements and estimated physical compensation
introduce uncertainty and dependence that must be assessed separately.
