# WLSScreenGNN

Auxiliary, read-only screening after balanced WLS and before additional
measurement acquisition. This implements the supplied v1 design: a phase
investigation score, a represented-anomaly score, and five independent family
scores (`hif`, `unbalance`, `measurement`, `parameter`, `topology`). Mixed
episodes can activate multiple families. Harmonic classification and
localization are outside v1; existing harmonic acquisition and diagnostics
remain available.

**Implementation status:** the model, numerical features, training,
calibration, evaluation, and optional episode integration are implemented.
A [five-seed practical IEEE-14 study](results/practical_v2_20260916/README.md)
now uses SFT-supported error mechanisms and explicit physical and measurement
criteria. On the same practical test cases, phase recall is 73.02% for V2 versus
60.78% for the frozen, recalibrated V1 model, with healthy trigger rates of
0.68% and 0.76%. These are conditional results for the selected population.
Weak-signal HIF coverage and cross-family decision thresholds remain major
limitations. It is not a dependable standalone error-type classifier.
The [original pilot](results/ieee14_v1_20260916/README.md) is retained separately.
The small training runs in unit tests remain software checks, separate from
the saved physical experiment.

The [reviewed fault scenario profile](../../docs/fault_scenario_review_20260917.md)
adds a versioned HIF curriculum, achieved-VUF strata, moderate-parameter and
harmonic sensitivity companions, and matched noise-accuracy views. Generate it
with `python -m research.reviewed_fault_scenarios --output-dir output/new_reviewed_cohort`.
Its physical generation and feature loading have been checked; it is not a new
trained-model result and does not overwrite the studies above.

Reviewed training now requires [WLS-observable expert admission](../../docs/wls_observable_training_20260917.md).
Use the generated `training_admission/<noise_profile>/filtered_manifest.jsonl`,
which freezes selected training observations and preserves held-out cases.
The training entry point rejects unfiltered reviewed means. Canonical LLM-SFT
expert-prefix files are separate from these GNN manifests; neither format
claims completed physical repair merely from a WLS alarm.

The bundle default now uses the [physical IEEE14 HIF profile](../../docs/ieee14_physical_hif_20260918.md):
69/13.8/18 kV bus bases, a 100–1000 ohm main population on 69 kV lines
(2.1–21.0 pu on the 69 kV base; 1000 pu there would be 47.6 kOhm, not a
1000 ohm HIF), and a separate voltage-stratified resistance sweep. Since
2026-09-19 the profile also carries the physical-ohm resistance classification
(`low_resistance_fault` < 50, `moderately_resistive` [50, 100),
`moderately_high_resistance` [100, 200), `representative_hif` [200, 500),
`weak_hif` [500, 1000), `extreme_weak_hif` [1000, 5000), `near_open_circuit`
>= 5000 ohm; every physical HIF row records `settings.hif_resistance_class`)
and an evaluation-only 69 kV detection-limit cohort (`hif_ohm_band:
"detection_limit"`, uniform 1000–5000 ohm, two slots per validation/test
parent, written to `hif_detection_limit_evaluation_manifest.jsonl`, never in
`split="train"`, and not part of the curriculum weights). The HIF injector now
refuses an omitted resistance on a multi-voltage registry; the historical 10 pu
default survives only on uniform normalized models. The prior normalized
profiles remain explicit reproduction options. The 7–8 voltage transition is
recognized as a transformer, while the graph dimensions and per-unit WLS
equations remain unchanged.

## Inputs and architecture

The only numerical inputs are the operator's configured MATPOWER bus/branch
model, the measured vector `[Vm, Pinj, Qinj, Pf, Qf, Pt, Qt]`, the actual
diagonal measurement covariance, and the fitted balanced-WLS state. `Vm`
retains its phase-A magnitude meaning; powers are total three-phase values
in system-base per unit. The graph never receives B/C voltages, phase currents,
sequence quantities, harmonics, simulator dispatch/load settings, hidden fault
nodes, scenario IDs, labels, or offline severity information.

- Bus features: **27**; directed branch features: **40**; global WLS features: **4**.
- Every registered branch contributes two directed edges, including configured
  open branches. Parallel circuits remain separate. Reverse edges swap terminal
  packets without negating their powers; tap/shift conventions remain native.
- Current-state Jacobian, actual covariance, signed normalized residuals,
  leverage, and unusable-residual masks are computed in float64. Cholesky solves
  and rank checks replace explicit inverses. Failed/unobservable solves return
  unavailable screens, never negative predictions.
- Three width-128 edge/node message-passing blocks, SiLU, residual connections,
  LayerNorm, dropout 0.1, mean/max aggregation and graph pooling; float32 Torch.
- A shared training-only scaler handles each measurement type across assets.
  Residual magnitudes are preserved. Inference uses the frozen scaler.
- Independent BCE losses have weights `1 / 0.25 / 0.1` for phase/anomaly/family.
  Unrepresented family heads are masked and omitted from inference reports.

The implementation uses plain PyTorch, without a PyTorch Geometric dependency.
Version 1 explicitly rejects partial measurement vectors, non-diagonal
covariance, constrained WLS, multiple reference buses, and invalid dimensions.
Masks are part of the fixed schema, but partial-observation WLS is not yet
implemented. The graph dimensions support IEEE-14 and IEEE-57; shared weights
do not establish transfer accuracy.

## Data contract

Use a JSONL manifest. `case` is a configured MATPOWER dictionary or a path to
its JSON representation; `z` is an array or JSON-array path. Paths are relative
to the manifest. Example row (put each complete object on one line):

```json
{
  "case": "configured_case.json",
  "z": "window_001.json",
  "parent_id": "independent_operating_parent_001",
  "window_id": "parent001_fault_variant002_window001",
  "families": ["hif", "measurement"],
  "severity": "weak",
  "split": "train",
  "measurement_convention": "phase_a_vm_total_3ph_power_net_injection_excludes_shunts"
}
```

`parent_id` must identify the shared physical operating parent **before**
healthy/faulty variants, overlays, and measurement noise are expanded. All
related variants stay in the same one of `train`, `validation`, `calibration`,
and `test`. Specify every split or omit every split to assign parents with the
fixed split seed. Splitting by episode IDs or noise draws is inappropriate.
The loader rejects conflicting assignments of the same parent.

An exhaustive `families: []` denotes a healthy label. A mapping such as
`{"hif": 1, "unbalance": null}` expresses partial labels; absent/null labels
are masked. Keep weak faults physically positive. Severity and other audit
attributes belong in `severity`/`offline_metadata` and never enter model inputs.

**Injection convention:** net injections exclude shunts already represented in
the configured admittance matrix. The legacy OpenDSS exporter currently adds
capacitor contributions to its injection vector. Do not label those old rows
as corrected. Regenerate or physically audit/correct the exports before
declaring the manifest convention above. The loader cannot infer this physical
provenance from numbers and intentionally does not guess a correction from
hidden truth.

For fresh noise expansion, additionally supply `measurement_kind:
"noiseless_mean"`, an explicit `measurement_sigma` array, `noise_seed`, and
`noise_replicates`. The loader samples the declared diagonal covariance and
reruns WLS for every noisy snapshot. Already noisy `observed` rows are never
noised again implicitly. Graph caches bind measurement windows, configured
cases, covariance, solver settings, and feature schema. Invalid graphs are
recorded separately and remain visible in coverage/availability reporting.

Generate a diverse corpus including corrected healthy controls, operating
changes, HIF strengths/locations, unbalance phases/patterns, competing balanced
errors, and mixed faults. The guide proposes 10,000–30,000 training graphs from
hundreds of independent parents, plus about 10,000 healthy calibration windows.
These are research starting points, not accuracy guarantees.

## Training and evaluation

For the practical SFT-supported profile, generate fresh physical data with:

```powershell
python -m research.gnn_screen.generate_practical_parallel --output-dir output/gnn/practical_corpus --seed 2026091933 --workers 4 --train-parents 384 --validation-parents 96 --calibration-parents 128 --test-parents 128 --healthy-validation-replicates 12 --healthy-test-replicates 40
```

The output includes a main manifest, a separate positive-label boundary manifest,
and proposal/physical-failure ledgers. The
[scenario decision](../../docs/gnn_practical_scenarios_20260916.md) explains the
offline paired-separation rule and its limits. Only the main manifest is used
for training; evaluate boundary coverage separately with the same frozen model
and calibration.

To train on the same HIF and unbalance windows as DAgger, build the corpus from
the DAgger physical corpora in `artifacts/measurements` instead:

```powershell
python -m research.gnn_screen.dagger_corpus --output-dir output/gnn_dagger_aligned/corpus --workers 16
```

Each DAgger window becomes one parent: HIF parents keep all ten scans with a
fault-removed healthy partner per scan on the HIF generator's own OpenDSS path,
and unbalance parents keep the stored unbalanced and balanced solves. Parameter,
dangling-terminal and meter errors follow DAgger's rules and are simulated on
the parent's path and operating point. Nothing is filtered by WLS
detectability; `offline_metadata.dagger_detectable` marks DAgger's training
subsets. The HIF resistance sweeps go to a separate test-only manifest
(`hif_resistance_evaluation_manifest.jsonl`). The defaults are the 2026-09-23b
HIF and unbalance corpora, simulated with generator reactive limits kept
(voltage regulation active) at OpenDSS solve tolerance 1e-8, so both simulator
paths converge to the same balanced solution; a corpus whose stored healthy
reference the current simulator cannot reproduce is refused, so earlier
corpora cannot be mixed in. The
`output/gnn_dagger_aligned_20260922` build and its results come from the
pre-fix, unregulated HIF corpora.

From the repository root, with Python, NumPy, SciPy, PyTorch, PyYAML and pytest:

```powershell
python -m research.gnn_screen.dataset manifest.jsonl --cache-dir output/gnn/cache --report output/gnn/corpus.json
python -m research.gnn_screen.train manifest.jsonl --config research/gnn_screen/config.yaml --cache-dir output/gnn/cache --output-dir output/gnn/run --device cpu
python -m research.gnn_screen.calibrate manifest.jsonl --checkpoint output/gnn/run/checkpoint.pt --cache-dir output/gnn/cache --output output/gnn/calibration.json
python -m research.gnn_screen.evaluate manifest.jsonl --checkpoint output/gnn/run/checkpoint.pt --calibration output/gnn/calibration.json --cache-dir output/gnn/cache --output output/gnn/evaluation.json
```

Training uses five seeds by default, AdamW, 64 graphs per batch, at most 100
epochs, patience 12, and gradient clipping at 1. Select the checkpoint on
validation phase recall at the declared healthy false-trigger operating point,
with non-phase trigger rate and validation loss as tie-breakers. Selection
never consumes calibration/test scores.

Freeze the selected checkpoint before calibrating thresholds using only healthy
calibration parents. A strict `score > healthy 99th percentile` is the default
empirical trigger policy, not a guaranteed future 1% false-trigger rate.
Calibration is bound to checkpoint bytes. Reports distinguish sigmoid scores
from posterior probabilities and report empirical healthy/non-phase triggers,
phase recall, family/severity strata, parent-bootstrap intervals, availability,
and the union with the declared WLS comparator. A union with WLS must not be
described using the GNN's standalone false-trigger rate.

Run strict IEEE-14-to-IEEE-57 transfer with frozen weights/scaler/threshold,
transfer with target healthy-only threshold calibration, and joint-system
training as separate studies. Matched-bank, pooled-MLP, raw-only, residual-only,
and mean-only ablations remain experiments to run; this implementation does
not claim that connectivity or any feature group has established added value.

## Episode integration

Enable only after training and independent threshold calibration:

```python
from psse_env.dagger.release_factories import production_environment_factory

env = production_environment_factory(
    screen_checkpoint="output/gnn/run/checkpoint.pt",
    screen_calibration="output/gnn/calibration.json",
)
```

The same options are available on `MatpowerDeploymentProviders`. They default
to disabled. Its successful `run_wls` returns `gnn_screen` evidence even when
the ordinary WLS alarm is negative. Expert and learner receive the same bound
report through the WLS context and tool history. Content bindings include the
measurement window, configured model, model identity, and current state.
The report also records the actual covariance hash and effective solver settings.

A positive phase trigger requests `get_three_phase_context`, followed by the
existing diagnostics when telemetry is available. A positive anomaly-only
trigger requests balanced contexts. Pending investigations block premature
finalization. The report does not alter WLS statistics, accept a correction,
certify healthy operation, estimate resistance/phase/distance, or replace
verification. Negative/unavailable screens retain independent coverage and
acquisition paths. Missing or incompatible model/calibration artifacts yield
an explicit unavailable report.

Standalone inference accepts only a configured case and the WLS vector:

```powershell
python -m research.gnn_screen.protocol_adapter --checkpoint output/gnn/run/checkpoint.pt --calibration output/gnn/calibration.json --case configured_case.m --measurements window.json --output output/gnn/screen.json
```

## Verification

```powershell
python -m pytest research/gnn_screen/tests -q
```

Tests cover actual IEEE-14/57 feature extraction, independent covariance and
Jacobian checks, custom covariance, unavailable evidence, hidden-input exclusion,
permutations, angle-reference invariance, parallel/open branches, terminal
reversal, batching and cross-graph isolation, mixed/masked labels, grouped
splits, training/calibration/evaluation, and episode routing. Synthetic pipeline
smoke tests establish executability only. Freeze the LLM for the first detector
comparison; targeted SFT/DAgger comes after the screen and threshold policy are
validated.

Validated locally after the practical V2 training run: **385 tests and 111 subtests passed**, covering
the GNN suite, shared numerical foundations, provider WLS, acquisition routing,
production protocol gates, and canonical protocol bridging. The recorded
pipeline roundtrip remains a software test. The separately saved physical study
includes independent corpus/metric audits and an actual trained-checkpoint
CPU/GPU inference comparison. These checks and its detection results are linked
from the [experiment report](results/practical_v2_20260916/README.md).
