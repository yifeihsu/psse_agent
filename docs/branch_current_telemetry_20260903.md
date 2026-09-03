# Per-Phase Branch-Current Telemetry for HIF and Unbalance Diagnosis

Date: 2026-09-03

## Decision

Per-phase terminal current phasors on every branch are now a first-class
telemetry channel (`three_phase_branch_currents`). With it, the faulted line,
phase, position, and resistance of a mid-span high-impedance fault have a
closed form, and the source bus of a three-phase load unbalance is
identifiable from Kirchhoff's current law. Both were open problems in the
voltage-only measurement set: the 2026-07-14 revision declared exact `alpha` a
no-go, and voltage-unbalance-factor ranking located the unbalanced load bus in
12% of the tracked corpus.

The tool protocol is unchanged. `run_three_phase_nlm_from_path` and both HIF
estimators keep their names and argument schemas; they behave differently only
when the channel is present in state metadata.

## A defect in the tracked imbalance corpus

While validating, the source of the 12% figure turned out to be the data, not
the method. The checked-in OpenDSS load file splits Bus 3 unevenly
(B3A/B3B/B3C at 0.5/0.3/0.2) for the original single-bus study. The imbalance
generator only *scaled* those loads, so every row of
`artifacts/measurements/out_measurements_imbalance` carries a second,
unlabeled unbalance at Bus 3.

| Corpus | Rows | Max-VUF bus is Bus 3 | Labeled Bus 3 | Median max VUF |
| --- | ---: | ---: | ---: | ---: |
| Tracked `out_measurements_imbalance` | 500 | 493 | 54 | 4.8% |
| New `out_measurements_imbalance_currents_20260903` | 220 | n/a (Bus 3 balanced) | 20 | 1.1% |

Consequences: the 2% `unbalance_vuf_threshold` in the deployment provider was
calibrated on the artifact (98.8% of old rows exceed it, 25.5% of new rows
do), and any unbalance-localization figure computed on the old corpus is
meaningless. The HIF generator was never affected because it already
rebalanced Bus 3. Both generators now rebalance Bus 3 before injecting their
own disturbance, and the imbalance meta records the override.

## Channel

```text
three_phase_branch_currents: one record per branch in MATPOWER order
  {branch, branch_row0, from_bus, to_bus, ibase_from_a, ibase_to_a,
   i_from_pu: [Ia, Ib, Ic], ang_from_deg, i_to_pu: [Ia, Ib, Ic], ang_to_deg}
branch_current_sigma_pu: declared per-component sensor sigma (default 1e-3 pu)
```

Currents flow *into* the branch from each terminal (the OpenDSS convention),
in per-unit on the 100 MVA base at the terminal bus. The exporter honors the
hidden split-line overrides used for mid-span HIF scenarios, so the external
branch identity is preserved. Sensor noise is independent Gaussian per real
and imaginary component. The HIF generator also stores a hidden
`three_phase_branch_currents_clean` copy per scan for QA replay; the scenario
generator strips it before the window reaches runtime metadata.

## Physics

**Unbalance source.** Summing the terminal currents into every branch at a bus
gives the per-phase current supplied by that bus's shunt elements. A balanced
constant-power load draws equal per-phase power whatever voltage unbalance it
sees, so the bus with the largest relative spread of per-phase shunt power is
the source. Negative-sequence voltage cannot do this: it peaks at electrically
weak buses. A per-bus noise floor (`6 * |V| * sigma * sqrt(n_incident)`) marks
spreads that noise alone could produce; insignificant buses rank below every
significant one, which stops unloaded buses from winning on a 0/0 ratio.

**HIF line and phase.** `|I_from + I_to - j(B/2)(V_from + V_to)|` is zero on a
healthy line and equals the fault current on the faulted phase. Over a
persistent scan window the complex differential is averaged coherently, so the
six-sigma detection floor falls as `6*sqrt(2)*sigma/sqrt(N)`.

**HIF position and resistance.** With both terminal voltages and currents, the
fault fraction `alpha` is where the fault-point voltage computed from either
end agrees (each segment keeping its share of the charging), and
`R = Re(V_x / I_fault)`. The two-terminal mismatch relative to the line drop
is a self-consistency check, and a fit at a terminal (`alpha` within 0.02 of
either end) is flagged ambiguous with a gross error on that terminal's sensor.

## Validation

Exporter physics, checked in `test_export_measurement_series.py` against
OpenDSS: per-phase shunt power from the currents matches the 122-entry bus
injections to `1e-3` pu at every bus; healthy-line differentials are below
`1e-8` pu; a hidden split-line HIF on Line.7-8 at `alpha=0.47`, 100 pu is
recovered to six decimals.

Closed-form accuracy on the new corpora (`scripts/validate_branch_current_localization.py`,
noise `sigma = 1e-3` pu applied):

| Unbalance, 220 rows | Value |
| --- | ---: |
| Source bus top-1 | 97.7% |
| Source bus top-3 | 99.5% |
| Top-1 significant against noise | 96.8% |
| Line-differential null quiet | 100% |
| Separation ratio median | 16.5 |

The five misses are rows where no bus clears the noise floor (a mild split at
a small load) and the ranking falls back to relative spread.

| HIF, 17 events x 10 diverse scans, R in [20, 200] pu | Per scan | Coherent window |
| --- | ---: | ---: |
| Line top-1 | 97.6% | 100% |
| Phase | 98.2% | 100% |
| Differential above 6-sigma floor | 62% | 100% |
| `alpha` abs error, median / p90 | 0.040 / 0.128 | 0.014 / 0.038 |
| R relative error, median / p90 | 7.2% / 24.2% | 2.3% / 8.5% |

For comparison, the 2026-07-14 benchmark's best condition (20 diverse scans,
voltage-only) had median `alpha` error 0.195 and 8% resistance error, and the
identical-noise control beat the diverse condition, which is why exact-`alpha`
generation was a no-go. The strict-physics dataset validator passes on the new
window corpus: 170/170 clean replays, bus balance and transformer signature
gates, zero issues.

Estimator behavior with the channel (`test_hif_multiscan_estimator.py`, real
OpenDSS): the single-scan search restricts to the detected phase, seeds from
the closed form, and recovers the exact candidate on a 3x3 grid. The
observability diagnostics now report the Cramer-Rao standard deviation of
`alpha` implied by the weighted residuals, per scan and for the whole window,
and identifiability requires the window value to sit inside the 0.05 audit
tolerance. Measured on Line.2-3 at `alpha=0.37`, phase B, `sigma = 1e-3` pu:

| Fault | Voltage-only, one scan | With currents, one scan | With currents, `alpha`-log-R correlation |
| --- | ---: | ---: | ---: |
| R = 100 pu | 2.12 | 0.097 | -0.07 (voltage-only 0.38) |
| R = 30 pu | 0.64 | 0.028 | -0.06 |

A spread of order one means the parameter is unobservable in practice, which
is what the July benchmark saw. With currents the spread scales as
`1/(I_fault * sqrt(N))`: a 100 pu fault needs about four scans to reach the
tolerance and ten scans reach roughly 0.03, consistent with the corpus figures
above; a 30 pu fault is identifiable from one snapshot. Windows whose scans
add no new sensitivity direction are still labeled `noise_averaging_only`
unless that averaging actually reaches the tolerance.

End to end through the deployment providers and the observable expert on the
first five windows and twelve unbalance rows of the new corpora (7x9 grid, 10
scans): every HIF window ran `run_three_phase_nlm_from_path` (terminal-current
method, correct line and phase, phase forwarded as `candidate_phase`) and the
multi-scan estimator (12 to 18 s each), accepted, and finalized as resolved.
One window's residual reduction was 0.187, below the 0.20 gate; it was
accepted on the terminal-current basis. All twelve unbalance rows localized
the labeled bus with a significant spread and finalized as resolved; under the
VUF-only gate six of the first eight had stalled unexplained.

## Sensitivity limits

- **Meter noise was applied to currents only.** Both corpora carry
  independent Gaussian noise of `1e-3` pu per real/imaginary component on
  every current phasor; the three-phase *voltage* phasors are the OpenDSS
  solution without noise, as in every earlier corpus (the estimators weight
  the voltage block at `5e-3` pu but the generators never applied it). Line,
  phase, and resistance are insensitive to this: they depend on the
  differential current and on `V_x`, which is of order 1 pu. The closed-form
  **position is not**. Its sensitivity to `alpha` is the voltage drop the
  fault current produces along the line, `|Z| * I_fault`, which is `1e-3` pu
  or less for a 100 pu fault, so voltage phasor noise competes with it
  directly. Synthetic check at `sigma_I = 1e-3`, 200 trials per cell,
  median `alpha` error of the single-scan closed form:

  | Line, `|Z|` (pu) | R (pu) | `sigma_V = 0` | `1e-4` | `1e-3` | `5e-3` |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | 1-2, 0.062 | 30 | 0.015 | 0.049 | 0.37 | 0.37 |
  | 1-2, 0.062 | 100 | 0.048 | 0.16 | 0.37 | 0.37 |
  | 2-3, 0.203 | 30 | 0.014 | 0.020 | 0.14 | 0.37 |
  | 2-3, 0.203 | 100 | 0.057 | 0.070 | 0.37 | 0.37 |
  | 9-14, 0.299 | 30 | 0.016 | 0.017 | 0.082 | 0.37 |
  | 9-14, 0.299 | 100 | 0.049 | 0.059 | 0.29 | 0.37 |

  An error of 0.37 is the uninformative level. Line and phase stayed at 100%
  and resistance error was unchanged in every cell. So the position results
  reported above assume voltage phasors accurate to about `1e-4` pu; at PMU
  class accuracy (`1e-2` pu) the closed-form position is unusable and the
  OpenDSS model fit, which also draws on how the fault current splits between
  the two terminals, is the only position evidence. Any corpus meant to
  qualify position claims must add voltage phasor noise.
- Current noise alone: at `sigma_I = 1e-3` pu the single-scan six-sigma floor
  is `8.5e-3` pu, so a single snapshot only *claims* detection for `R` below
  about 118 pu; the line still ranks first in 97.6% of scans. Ten coherent
  scans lower the floor to `2.7e-3` pu (R up to about 370 pu). Position error
  from current noise is roughly `sigma_I / I_fault` per scan (0.1 at 100 pu)
  and averages down over the window.
- A fault fitted within 0.02 of a terminal is not accepted on differential
  evidence alone, because a gross error on that terminal's current sensor
  produces the same phasors. The OpenDSS model fit remains the arbiter there.
- The differential test assumes the diagonal (uncoupled) line model of the
  checked-in OpenDSS case; mutual coupling would need the full phase matrix in
  the charging term.

## Pipeline integration

- `TransactionalPSSEEnv` lists the channel in `available_evidence`, and the
  production-row telemetry gate accepts it in place of a stored NLM diagnostic.
- `DiagnosticsExpert` routes to the NLM tool on the channel alone and forwards
  the observed `suspected_phase`.
- The deployment `run_three_phase_nlm_from_path` uses the coherent window when
  a scan window carries currents; the stored legacy diagnostic is reported as
  `legacy_nlm_top_branch_row0` only.
- HIF acceptance gains `acceptance_basis`: `residual_reduction_vs_null` as
  before, or `terminal_current_differential` when the differential is detected,
  the fault impedance is positive and resistive, both terminals agree, the fit
  is interior, and the model search agrees on the phase.
- Unbalance acceptance with the channel requires a noise-significant source and
  a quiet line-differential null; `voltage_gate_passed` is reported, not
  decisive. The explanation carries `bus_1based` for the release audit.
- The scenario generator propagates the channel into `metadata`,
  `hif_runtime`, and every scan of `hif_scan_window`, and derives a balanced
  current control for `telemetry_no_disturbance`.
- The generator's default unbalance corpus is now the bus-3-rebalanced
  branch-current corpus; the defective 2026-07 corpus remains reachable only
  as `LEGACY_IMBALANCE_SAMPLE_PATH`. The frozen BC0 HIF defaults stay
  voltage-only; the branch-current HIF corpora are exposed as
  `CURRENT_TELEMETRY_HIF_SAMPLE_PATHS` for the research path below.

## Research DAgger integration

Added later on 2026-09-03 so the diagnostic families can enter the
learner-in-the-loop research round without touching any frozen release
input.

**VUF gate recalibrated to 1%.** The 2% gate had been calibrated on the
defective corpus. With bus 3 balanced, the balanced telemetry control has VUF
exactly 0, the 170 HIF scans of the 17-window corpus peak at 0.32%, and the
220 corrected unbalance rows have median 1.1% (25.5% clear 2%, 53.2% clear
1%). One percent keeps a 3x margin above the strongest HIF-induced VUF, so the
flag never fires on a pure HIF row. The constant lives in
`three_phase_nlm/branch_current_analysis.py` and is shared by the deployment
provider and the scenario generator.

**Honest unbalance signatures.** The generator no longer stamps
`three_phase_unbalance vuf_threshold_exceeded` on every row. It emits that
flag only when the row's largest bus VUF clears the shared gate, and a second
flag, `three_phase_unbalance phase_current_spread_detected`, only when the
branch-current channel exposes a noise-significant unbalance source with a
quiet line-differential null. A row exposing neither is rejected as
`unbalance_not_observable` (about 3% of the corrected corpus). Both strings
carry the family marker, so routing and explained-anomaly closure are
unchanged; only the policy-visible text is now true of the telemetry.

**Training corpus for HIF.** Seventeen windows cannot cover training,
development, and a protected holdout, so
`artifacts/measurements/hif_multiscan_currents_train_85x10_20260903/` adds 85
windows (seed 20260904, five per eligible line, ten diverse scans each,
resistance 20-200 pu, currents at `sigma = 1e-3` pu). Strict-physics QA
replayed all 850 snapshots with zero issues. Closed-form accuracy on it:

| HIF, 85 windows x 10 scans | Per scan | Coherent window |
| --- | ---: | ---: |
| Line top-1 | 95.7% | 100% |
| Phase | 96.7% | 100% |
| `alpha` abs error, median / p90 | n/a | 0.016 / 0.048 |
| R relative error, median / p90 | n/a | 3.6% / 11.9% |

**Research entry point.** `scripts/run_dagger_research.py` gained
`--plan-preset {core,diagnostic,combined}`. The `diagnostic` preset trains on
12 HIF, 6 measurement+HIF, 12 unbalance, and 6 balanced-telemetry-control
roots with a 6/3/6/3 development split; `combined` unions it with the
correction-family round. Diagnostic plans default to the branch-current
corpora (`--hif-sample-paths`, `--imbalance-sample-path` override) and keep
every scan of a ten-scan window. `--hif-search-profile auto` selects the
research OpenDSS budget (7x9 grid over ten scans, the configuration validated
above) whenever the plan contains an HIF family, through a research-only
environment factory that is otherwise identical to the production one; the
release factory module stays content-pinned and unmodified. The resolved
profile is recorded in the run config and only participates in the resume
check when it differs from the legacy core configuration, so earlier runs
still resume.

**Counterfactual ladder.** `plausible_wrong_actions` now injects diagnostic
mistakes whenever the expert proposes a diagnostic tool: estimating before
localizing, running the wrong family's diagnostic, estimating on a healthy
line, overrunning the bounded search budget, escalating before the configured
estimators ran, and applying a parameter correction that would mask an
explanation-only anomaly. The healthy-line and right-line targets are
placeholders that the counterfactual generator binds from the injected
branch's own NLM output after its setup action, never from hidden truth.

**Expert-only end-to-end check** through the research environment factory on
a diagnostic-preset build (seed 20260905; 3 HIF, 2 measurement+HIF, 4
unbalance, 2 control roots, none rejected): every HIF and measurement+HIF root
ran localization then the multi-scan estimator, was accepted on the
branch-current basis, and finalized with the correct line and phase in 14-18 s
per episode (resistance error 0.3-4.6%); the four unbalance roots localized
the labeled bus and finalized in under a second, two carrying both sensor
flags and two only the current-spread flag; both controls finalized after one
WLS pass with no explanation. The counterfactual ladder produced eleven rows
per root (63 s on the HIF root, 7 s on the unbalance root). Every diagnostic
injection either failed observably (`hse_runtime_missing`,
`hif_search_budget_invalid`, `hif_diagnostic_ladder_incomplete`,
`operator_escalation_not_supported`, `correction_route_not_actionable`) or
ran a rejected fit on the wrong line, and the expert supplied a recovery
target in every case. One observation for later expert review: after the
masking parameter correction is refused on an HIF root, the rank-one recovery
is the next classical context request rather than a return to the diagnostic
ladder, because the injected parameter-context request has already primed the
classical experts on the WLS-inconsistent operator vector.

**Still outside this change.** A BC0 freeze that admits unbalance and the
current-telemetry HIF corpus (new suite quotas, family-policy floors, a v2
study manifest) needs an expert-baseline run on the new roots first; the
DAgger-1 release collector's family-keyed schedule constants are deliberately
untouched. A measurement+unbalance composition was not added: the 122-entry
operator vector of an unbalanced OpenDSS solve is not consistent with the
balanced positive-sequence WLS model (every corrected unbalance row and every
HIF window fails the case14 chi-square test on its own), so an overlaid bad
meter is not separable by the fundamental-frequency residuals; measurement+HIF
already carries the same caveat under its explicit handoff allowance.

## Artifacts

- `artifacts/measurements/out_measurements_imbalance_currents_20260903/` (220 rows, `branch_current_localization_report.json`)
- `artifacts/measurements/hif_multiscan_currents_17x10_20260903/` (17 x 10 scans, `quality_report.json`, `branch_current_localization_report.json`)
- `artifacts/measurements/hif_multiscan_currents_train_85x10_20260903/` (85 x 10 scans, seed 20260904, `quality_report.json`, `branch_current_localization_report.json`)
