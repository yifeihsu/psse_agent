# Multi-Scan HIF Parameter Estimation

The multi-scan HIF path estimates parameters that persist across an event
window while replaying a separate operating point for every scan.

The active sequence is:

```text
wls_from_path
  -> run_three_phase_nlm_from_path
  -> estimate_hif_location_magnitude_multiscan_from_path
```

The NLM stage selects the suspected line. The multi-scan stage estimates the
line fraction `alpha`, phase, and HIF resistance.

## Per-Phase Branch-Current Telemetry (2026-09-03)

The 2026-07-14 revision found that the 122-entry operator vector plus bus
voltages cannot identify `alpha`. The recommended fix, per-phase terminal
current phasors on the lines, is now a first-class channel:

```text
three_phase_branch_currents: [
  {branch, branch_row0, from_bus, to_bus, ibase_from_a, ibase_to_a,
   i_from_pu: [Ia, Ib, Ic], ang_from_deg: [...],
   i_to_pu:   [Ia, Ib, Ic], ang_to_deg:   [...]}, ... 20 branches]
branch_current_sigma_pu: declared per-component sensor sigma (default 1e-3)
```

Every phasor is the current flowing *into* the branch from that terminal, in
per-unit on the 100 MVA base at the terminal bus. The exporter is
`IEEE_14_OpenDSS.export_measurement_series.extract_three_phase_branch_current_measurements`;
the analysis lives in `three_phase_nlm/branch_current_analysis.py` and needs no
OpenDSS.

What the channel changes:

- **Line and phase come from telemetry.** `run_three_phase_nlm_from_path`
  ranks lines by the per-phase differential current
  `|I_from + I_to - j(B/2)(V_from + V_to)|` (charging removed) and names the
  phase carrying it. With a persistent scan window it averages the complex
  differential across scans, so the six-sigma detection floor falls as
  `6*sqrt(2)*sigma/sqrt(N)`. Method: `terminal_current_differential`.
- **Position and resistance have a closed form.** With both terminal
  voltages and currents, `alpha` is the point where the fault-point voltage
  computed from either end agrees, and `R = Re(V_x / I_fault)`. The
  estimators report it as `terminal_current_estimate`, seed the OpenDSS grid
  search around it, add a current residual block to the objective, and
  restrict the phase search when the differential singles out one phase.
- **Observability is quantified.** The observability payload reports the
  Cramer-Rao standard deviation of `alpha` per scan
  (`single_scan_alpha_std_min`) and for the window
  (`alpha_crlb_std_effective`); `parameter_identifiable` requires the window
  value inside the 0.05 audit tolerance. Voltage-only scans give a spread of
  order one (unobservable in practice); currents cut it by more than an order
  of magnitude, so a window whose scans only average noise is labeled
  `noise_averaging_only` unless that averaging reaches the tolerance.
- **The expert passes the phase.** `DiagnosticsExpert` forwards the NLM
  tool's `suspected_phase` as `candidate_phase`, bounding the estimator search.

Generate windows with currents (noise applied as `sigma * --noise-scale`):

```bash
python Transmission/generate_measurements_hif_ieee14.py \
  --out artifacts/measurements/hif_multiscan_currents_17x10_20260903 \
  --n-hif 17 --n-no-error 0 --seed 20260903 \
  --scans-per-window 10 --operating-point-mode diverse \
  --noise-scale 1.0 --branch-current-noise-pu 0.001
```

Score the closed form against the hidden labels:

```bash
python scripts/validate_branch_current_localization.py \
  artifacts/measurements/hif_multiscan_currents_17x10_20260903/samples.jsonl
```

Results and sensitivity limits are in `docs/branch_current_telemetry_20260903.md`.

## Parameter Model

The default `resistance_mode=shared` estimates:

```text
shared: alpha, phase, log(R_hif)
scan-specific: solved OpenDSS network state and operating point
```

`resistance_mode=scan_specific_smooth` keeps alpha and phase shared, profiles a
separate log resistance per scan, and penalizes adjacent resistance changes.
This mode reports the median and range in addition to per-scan resistance.

The estimator minimizes the robust joint residual across the selected scans.
The 122-entry operator vector and three-phase voltage phasors are weighted as
separate residual blocks. The default robust loss is `soft_l1`.

Each distinct operating point is solved from a freshly compiled circuit. This
is intentional: reusing one active OpenDSS circuit across diverse scans leaves
Model=3 generator reactive-power iteration state behind and changes the
candidate residual. An in-memory cache still reuses exact simulations for the
same `(phase, alpha, R_hif, op_point)` key, including identical-noise scans and
repeated local-optimizer evaluations.

## Operating-Point Contract

Every scan contains an `op_point` object. Supported fields are:

```json
{
  "load_scale": 1.0,
  "bus_load_scales": {"b2": 0.95, "b4": 1.08},
  "generator_dispatch_kw": {"b2": 43000.0},
  "voltage_setpoints_pu": {"b2": 1.04, "b6": 1.065},
  "source_voltage_pu": 1.058
}
```

`load_scale` is the event-wide multiplier. Each `bus_load_scales` value is a
spatial profile factor multiplied by `load_scale`; all load buses are explicit
in the canonical form. Generator dispatch values are absolute kW, and voltage
values are absolute per-unit setpoints. Generation and estimation both apply
these fields through `three_phase_nlm/hif_operating_point.py`.

The topology and HIF label must remain fixed inside one window. The generator
keeps scan 0 at the reference global load scale so the existing NLM bridge can
localize the line against a matching model.

## Generate Event Windows

Ten identical-operating-point scans with independent noise:

```bash
python Transmission/generate_measurements_hif_ieee14.py \
  --out artifacts/measurements/hif_multiscan_identical_10 \
  --n-hif 100 \
  --n-no-error 0 \
  --scans-per-window 10 \
  --operating-point-mode identical_noise \
  --noise-scale 1.0
```

Ten diverse scans with one persistent HIF:

```bash
python Transmission/generate_measurements_hif_ieee14.py \
  --out artifacts/measurements/hif_multiscan_diverse_10 \
  --n-hif 100 \
  --n-no-error 0 \
  --scans-per-window 10 \
  --operating-point-mode diverse \
  --noise-scale 1.0
```

Use `--scans-per-window 20` for the 20-scan saturation experiment. Generated
rows retain top-level `z_obs` and `three_phase_voltages` from scan 0 for
backward compatibility, and add `scans`, `scan_count`, and `shared_label`.
Each scan also stores a hidden `z_clean` QA vector for deterministic replay;
the estimator and visible SFT trace must not consume that field.

Validate schema, physical consistency, and clean replay before estimation:

```bash
python scripts/validate_hif_multiscan_dataset.py \
  artifacts/measurements/hif_multiscan_diverse_10/samples.jsonl \
  --meta artifacts/measurements/hif_multiscan_diverse_10/meta.json \
  --strict-physics \
  --output artifacts/measurements/hif_multiscan_diverse_10/quality_report.json
```

## Validate Parameter Recovery

```bash
python scripts/validate_hif_multiscan_parameter_estimates.py \
  --samples artifacts/measurements/hif_multiscan_diverse_10/samples.jsonl \
  --scan-count 10 \
  --max-scans 10 \
  --scan-selection information_greedy \
  --alpha-grid-size 31 \
  --r-grid-size 35 \
  --enforce-production-gates
```

Run the same command with `--scan-count 1`, an identical-noise dataset, and
5-, 10-, and 20-scan diverse conditions. The production trace builder retains
up to 20 candidate scans and selects 10 by default; this is the current
accuracy/runtime tradeoff, not evidence that alpha is identifiable.

Production gates cover line top-1 accuracy, alpha and resistance error, top-k
truth coverage, false precision, near-best alpha interval coverage, effective
rank, condition number, and alpha-log-resistance correlation.

## Observability Diagnostics

For every selected scan, the estimator computes finite-difference
sensitivities in `[alpha, log(R_hif)]` after solving the OpenDSS power flow. It
then sums the reduced per-scan information matrices. In scan-specific
resistance mode, resistance contrasts are profiled out after adding the
smoothness regularizer.

The diagnostics report:

```text
effective_rank
smallest_singular_value
condition_number
alpha_log_r_correlation
information_gain_vs_best_single_scan
scan_diversity_score
status
parameter_identifiable
```

If a finite-difference perturbation fails for an isolated scan, shared-
resistance mode reports `diagnostic_partial`, retains information from the
successful scans, and forces `parameter_identifiable=false`. This avoids
discarding all diagnostics without permitting a point-location claim from an
incomplete numerical check.

This is a reduced finite-difference information calculation at known operating
points. It is not presented as a full free-state WLS Schur complement. A
successful OpenDSS solve can still return `parameter_identifiable=false`.

Point location should be reported only when the fit is non-ambiguous and the
observability gate passes. Otherwise the trace reports the near-best alpha and
resistance intervals.

The corrected 17-event benchmark did not pass the alpha or top-k production
gates. Current `noise_averaging_only` and `diagnostic_partial` outputs therefore
must not expose point alpha. See `docs/hif_multiscan_revision_20260714.md` for
the measured results and no-go decision.

Label-backed mock estimates are smoke-test fixtures. They are tagged
`synthetic_oracle` and fail `scripts/validate_hif_traces.py` unless the explicit
`--allow-mock-estimator` override is supplied.
