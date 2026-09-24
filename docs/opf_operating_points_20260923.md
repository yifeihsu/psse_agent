# OPF-driven operating points for the OpenDSS corpora (2026-09-23)

Branch `codex/scada-pmu-evidence`, work package 4. Companion of
[ieee14_hif_legacy_reconfiguration_20260919.md](ieee14_hif_legacy_reconfiguration_20260919.md), whose
`*_20260923b` corpora this revision supersedes as `*_20260923opf`.

## Why

A fairness audit of the twelve scenario families found that balanced SCADA alone separates the four
OpenDSS-generated families (`hif`, `measurement+hif`, `three_phase_unbalance`,
`telemetry_no_disturbance`) from the eight pypower-generated ones. The pypower families sit on the
case14 AC-OPF dispatch at their load scale: the units at buses 3, 6 and 8 run (up to 53, 12 and 32 MW at
load scale 1.25), bus 8 is held at its 1.06 pu upper bound, and the measured bus-3 active injection is
-0.79..-0.62 pu. The OpenDSS corpora ran the checked-in model dispatch: bus 2 at 40 MW, the three other
units as 1 kW synchronous condensers, the file setpoints (bus 8 at 1.09), so bus 3 injected
-1.19..-0.75 pu and bus 8 sat at 1.073-1.099 pu after the reactive-limit fix. Two nearly disjoint
ranges in plain SCADA are a family cue no policy should be given.

The operating-point mechanism (`three_phase_nlm/hif_operating_point.py`) already accepted per-unit
active dispatch, PV voltage setpoints and a source voltage. This revision fills those fields from the
same AC-OPF the pypower families use, at the same per-bus loads, and regenerates the HIF and unbalance
corpora with unchanged seeds.

## Dispatch law

`three_phase_nlm/hif_operating_point.py` gains the OPF side of the operating point:

- `scaled_case14(load_scale, bus_load_scales)` builds pypower `case14()` with `PD`/`QD` at every bus
  multiplied by `load_scale * bus_load_scales[bus]`, the exact factor `apply_hif_operating_point`
  gives the OpenDSS load at that bus (the IEEE14Loads.DSS loads are the case14 loads in kW).
- `solve_ieee14_opf(...)` runs `runopf(case, ppoption(VERBOSE=0, OUT_ALL=0))`: the same call, cost
  table (case14 `gencost`), ratings (RATE_A 9900 MVA, so no thermal constraint binds) and voltage bounds
  (0.94/1.06) as `Round0ScenarioGenerator._solve_ac_opf` and `Transmission.generate_measurements`.
- `operating_point_from_opf_solution(...)` returns the canonical operating point: `generator_dispatch_kw`
  is the OPF `PG` of the units at buses 2, 3, 6 and 8 (kW; an interior-point zero below 1e-6 MW is
  written as 0 kW), `voltage_setpoints_pu` their OPF bus voltage magnitudes, `source_voltage_pu` the
  OPF voltage magnitude at bus 1. The slack is folded into the source: the OpenDSS Vsource holds the
  OPF bus-1 voltage and supplies whatever active and reactive power the network needs, which in a
  converged solve is the OPF slack output (the bus-1 injections agree to 1e-6 pu).
- `ieee14_opf_operating_point(load_scale, bus_load_scales)` combines the three and raises
  `OPFDispatchError` when the OPF does not converge or returns an invalid dispatch. There is no
  fallback path; callers skip the window and record it.
- `apply_ieee14_dispatch_and_setpoints(baseline, op_point)` is the generator-and-source half of
  `apply_hif_operating_point`, split out so the unbalance generator can scale and split its own loads
  and still apply the dispatch. `apply_hif_operating_point` calls it after scaling the loads and keeps
  its behaviour and return value (old corpora replay bit-identically; the reactive-limit restoration is
  unchanged).

A 0 kW `Model=3` generator still regulates its bus in OpenDSS (checked: at load scale 0.8 the OPF idles
the three units and bus 8 supplies 5.6 MVAr at 1.06 pu). The OpenDSS model reproduces the pypower OPF
solution at every operating point tried: the 122-entry measurement vectors differ by at most 0.02
measurement sigma (9e-6 pu in Vm, 1e-6 pu in P, 2e-4 pu in Q, the latter the Vsource's finite
short-circuit impedance).

## The HIF generator (`Transmission/generate_measurements_hif_ieee14.py`)

`--dispatch-mode opf|case14` (default `opf`). For every window the event stream draws the load scale,
line, split ratio, phase and resistance exactly as before. The scan stream then draws, per diverse scan,
the spatial load profile, the case14-mode bus-2 dispatch perturbation and the setpoint perturbations, in
the same order in both modes; in `opf` mode the dispatch and setpoint draws are discarded and
`ieee14_opf_operating_point(load_scale, bus_load_scales)` supplies the dispatch, the PV setpoints and
the source voltage of that scan. Scan 0 is the OPF at the uniform event load scale, which is also the
row's `z_reference_opf`, so that vector is now the same operating point as `z_true` (the balanced OpenDSS
solve at scan 0's point, fault removed) instead of a differently dispatched case.

The stored `op_point` stays the canonical five-key schema that `scripts/validate_hif_multiscan_dataset.py`
requires and that every replay path canonicalizes (`_simulate_base`, `_simulate_candidate`, the
multiscan estimator, the audit's paired healthy solve, the GNN corpus converter); the OPF values are in
its `generator_dispatch_kw`, `voltage_setpoints_pu` and `source_voltage_pu`. The provenance sits beside
it: every scan and every row carry a `dispatch` block (`mode`, solver, OPF objective, slack P/Q, unit
reactive outputs), `window_metadata.dispatch_mode` names the mode, and `meta.json` records it under
`hif.dispatch`, `hif.scan_window.dispatch_mode` and `hif.generation.dispatch_mode`. Healthy controls are
pypower OPF rows in both modes and carry the same block.

An OPF that does not converge for any scan skips the whole window: the window is listed in
`hif.generation.skipped_windows` (id, index, load scale, line, reason), a warning is printed, and no
row is written. Because the operating points come from their own seeded stream, the labels of every
other window are unchanged. A skipped healthy control is listed in `skipped_controls`.

`--three-phase-noise-pu` and `--branch-current-noise-pu` already reached the generator; `meta.json`
(`three_phase_sigma`, `branch_current_sigma_pu`, `noise_contract`, and now
`hif.generation.three_phase_noise_pu` / `branch_current_noise_pu`) and every row and scan declare the
applied sigmas. The phasor sigmas scale the same standard-normal draws, and the SCADA noise of each scan is
drawn before the phasor noise, so changing them leaves every SCADA observation bit-identical.

## The unbalance generator (`Transmission/generate_measurements_imbalance.py`)

`--dispatch-mode opf|case14` (default `opf`). After the window's load scale, phase split and target bus
are drawn, `ieee14_opf_operating_point(load_scale)` is solved once; its solution is `z_reference_opf`,
and its dispatch is written with `_apply_operating_point_dispatch` after the loads are scaled and split
(unbalanced solve) and after the loads are scaled (balanced `z_true` solve). The row `op_point` keeps
`load_scale` and `target_bus` first and then carries the canonical dispatch keys, so
`canonicalize_ieee14_operating_point(row["op_point"])` replays the applied dispatch in the scenario
generator's balanced-reference solve; a `dispatch` block and `meta.json` (`imbalance.dispatch`,
`imbalance.dispatch_mode`, `imbalance.generation`) record the provenance. `_apply_operating_point_dispatch`
is a no-op for a load-only op_point, so the pre-opf corpora replay unchanged through the same code. A
non-converged OPF skips the window into `imbalance.generation.skipped_windows`.

## Regeneration script

`scripts/regenerate_hif_physical_corpora.py` forwards `--dispatch-mode` (default `opf`),
`--three-phase-noise-pu` / `--branch-current-noise-pu` (HIF, default 1e-4 / 1e-4) and
`--unbalance-three-phase-noise-pu` / `--unbalance-branch-current-noise-pu` (default 5e-3 / 1e-3) to the
generators, records them in the receipts and the implementation manifest (`family_flags`, per-recipe
arguments), and its strict unbalance replay applies each row's stored dispatch. A corpus short of its
recipe because of skipped windows stops the run with the receipt naming the exclusions.

## Regeneration: `*_20260923opf`

`python scripts/regenerate_hif_physical_corpora.py --tag 20260923opf --output-dir output/hif_physical_revision_20260923opf --families hif,unbalance --workers 6`
(defaults: `--dispatch-mode opf`, HIF phasor sigmas 1e-4 / 1e-4, unbalance 5e-3 / 1e-3), commit 4161bb8 plus this
revision's working-tree edits (recorded in the implementation manifest). Receipts, logs, replay reports, audits
and the manifest are under `output/hif_physical_revision_20260923opf/`.

| corpus | windows x scans | controls | generation | skipped |
|---|---|---|---|---|
| `hif_physical69_main_train_84x10_20260923opf` | 84 x 10 | 20 | 331 s | 0 |
| `hif_physical69_main_valid_21x10_20260923opf` | 21 x 10 | 5 | 88 s | 0 |
| `hif_physical69_detection_limit_21x10_20260923opf` | 21 x 10 | 0 | 86 s | 0 |
| `hif_physical_sweep_eval_336x10_20260923opf` | 336 x 10 | 0 | 1149 s | 0 |
| `hif_physical69_main_train_extra_252x10_20260923opf` | 252 x 10 | 60 | 931 s | 0 |
| `hif_physical69_main_valid_extra_63x10_20260923opf` | 63 x 10 | 15 | 247 s | 0 |
| `out_measurements_imbalance_currents_ybus_440_20260923opf` | 440 | 60 | 110 s | 0 |

The six HIF corpora ran in parallel (the OPF adds about 0.2 s per scan, 7,770 OPF solves in all); no OPF failed,
so no window or control was skipped. Two OpenDSS solves diverged on their first attempt (one in the 84x10 train
corpus, one in the 21x10 valid corpus, the known once-per-several-thousand-solves engine behaviour) and converged
on the fresh compile; the tolerance stays 1e-8. Every validator passed: the sample validator with the same
legacy-NLM top-1 ranking misses as before (4 in the sweep, 1 in the training extra, all targets in the top 3),
the strict-physics multiscan replay reconstructs all 7,770 clean scans to 1e-9 through
`_simulate_candidate` at the stored op_points, the branch-current localization report is unchanged in kind,
and the unbalance replay (balanced reference and unbalanced sensor mean of all 440 rows, dispatch applied)
is exact (max error 0.0).

## WLS admission before and after (margin 1.25)

The admitted windows are the same windows. The discovered-mode audit (chi-square 0.01 OR normalized residual
4.0, anomaly margin 1.25, reference scan) admits 27 + 8 + 77 + 19 = 131 HIF windows as in 20260923b, and the
admitted ids are identical corpus by corpus. Reference-scan alarms move slightly with the dispatch (39, 11,
122, 30 against 39, 11, 126, 31 before), healthy-control alarms are unchanged (1 of 20, 0 of 5, 1 of 60,
0 of 15; paired healthy alarms 1, 0, 6, 1; sweep 6 of 336), and the detection-limit corpus still raises no
alarm. The unbalance corpus admits 162 of 440 windows (160 before; 157 in common, 5 newly admitted, 3 no
longer, the two `unbalance_not_observable` rejections against three): 218 alarms at margin 1.0 as before, 2 of
60 controls alarm at margin 1.0 and 0 at margin 1.25, as before.

Detectable subsets (every control kept, admitted windows copied verbatim; the re-audits admit every window):
`hif_physical69_main_train_detectable_27x10_20260923opf` (27 windows, 20 controls),
`hif_physical69_main_valid_detectable_8x10_20260923opf` (8, 5),
`hif_physical69_main_train_extra_detectable_77x10_20260923opf` (77, 60),
`hif_physical69_main_valid_extra_detectable_19x10_20260923opf` (19, 15) and
`out_measurements_imbalance_currents_ybus_detectable_162_20260923opf` (162, 60).

## What changed and what did not, against 20260923b

Row by row (same ids, same order) over the six HIF corpora (777 windows, 7,770 scans) and the unbalance corpus
(440 windows), checked by `scratchpad/wp4/compare_corpora.py`:

- Fault labels are identical in every window: line, phase, resistance (ohm and pu), split ratio, resistance
  class, and the event load scale. Unbalance labels (bus, phase fractions, load scale) are identical too.
- Every scan's load profile (`load_scale`, `bus_load_scales`) is identical, because the case14-mode dispatch and
  setpoint draws are still consumed in opf mode.
- The SCADA noise of every scan (`z_obs - z_clean`) is bit-identical, and every healthy control row is
  bit-identical (`z_obs` equal): the phasor sigma change and the dispatch change touch neither.
- What differs is the dispatch and therefore the clean physics: all 7,770 scans carry `dispatch.mode = opf`,
  the PMU phasors of the HIF corpora are drawn at 1e-4 per component (`three_phase_sigma`,
  `branch_current_sigma_pu` and the `noise_contract` declare 1e-4 in `meta.json`, every row and every scan;
  the unbalance corpus keeps 5e-3 / 1e-3).

Dispatch statistics of the healthy reference (`z_true`, scan 0's operating point with the fault removed),
min / median / max, against an OPF root sample built with `Round0ScenarioGenerator._solve_ac_opf(
_scaled_case14(load_scale))` at the very load scales of the windows (the pypower families' law):

| quantity | 20260923b (case14 dispatch) | 20260923opf | OPF root sample |
|---|---|---|---|
| HIF, bus-3 P injection (pu) | -1.176 / -0.979 / -0.754 | -0.773 / -0.648 / -0.641 | -0.773 / -0.648 / -0.641 |
| HIF, bus-8 Vm (pu) | 1.081 / 1.090 / 1.090 | 1.060 / 1.060 / 1.060 | 1.060 / 1.060 / 1.060 |
| HIF, bus-8 Q injection (pu) | 0.142 / 0.183 / 0.240 | 0.056 / 0.087 / 0.118 | 0.056 / 0.087 / 0.118 |
| unbalance, bus-3 P injection (pu) | -1.177 / -0.954 / -0.754 | -0.773 / -0.652 / -0.641 | -0.773 / -0.652 / -0.641 |
| unbalance, bus-8 Vm (pu) | 1.081 / 1.090 / 1.090 | 1.060 / 1.060 / 1.060 | 1.060 / 1.060 / 1.060 |

Window by window the OpenDSS reference and the pypower OPF root agree to better than 5e-7 pu on both channels
(the tabular corpus's measured -0.79..-0.62 and 1.046..1.064 are these values plus SCADA noise). The bus-3
and bus-8 cue that separated the OpenDSS families is gone; what remains between the simulators at the same
operating point is the 2e-4 pu reactive offset of the Vsource's finite short-circuit impedance (0.02 SCADA
sigma, but 2 sigma at PMU precision, see below).

## PMU precision 1e-4 and the estimator's acceptance gate

The HIF corpora's three-phase voltage and branch-current phasors are drawn at sigma 1e-4 per real/imaginary
component (before: 5e-3 / 1e-3); `meta.json`, every row and every scan declare `three_phase_sigma = 1e-4`,
`branch_current_sigma_pu = 1e-4` and the matching `noise_contract` channels, so the estimators weight the
phasors exactly as they were drawn. The unbalance corpus keeps 5e-3 / 1e-3.

The multiscan estimator (`estimate_hif_location_magnitude_multiscan`, shared resistance, all 10 scans) was run
on the 27 windows of the two valid detectable subsets and its result put through the provider's unchanged
fail-closed gate (`MatpowerDeploymentProviders._hif_diagnostic_acceptance`: weighted residual norm at most
3.0, residual reduction against the no-HIF null at least 0.20 or a conclusive terminal-current differential,
no model mismatch). All 27 are accepted on `residual_reduction_vs_null`: the HIF fit's weighted residual norm
is 0.972-1.026 (median 1.00, i.e. the noise floor; the model replays the scans exactly), the no-HIF null
norm is 108-383 (median about 230), the reduction 0.991-0.997, every window `full_rank_well_conditioned`,
the estimated phase correct in 27 of 27 and the estimated split ratio within 0.001 of the label. The gate was
not loosened. Each window takes about 150 s at this setting; the 104 training-subset windows were not run
(`scratchpad/wp4/gate/*.jsonl`, `gate_runner.py`).

## Tests

`tests/test_opf_operating_points.py` (new, 12 tests): the OPF operating point is reproduced by
`_simulate_base` (0.02 sigma against the pypower solution) with bus 8 at its OPF voltage; the slack is folded
into the source (Vsource pu equals the OPF bus-1 voltage, the bus-1 injection equals the slack output); 0 kW
units regulate; the scan sampler keeps the load profiles of a seed and takes the OPF at each scan's loads;
case14 mode reproduces the stored 20260923b op_points bit for bit; tiny-seed generation in both modes keeps
labels, load profiles and SCADA noise identical while declaring 1e-4 and replaying exactly through the
estimator simulator and the strict validator; the unbalance generator keeps labels and replays the stored
dispatch exactly (and the scenario generator's canonicalized replay lands on `z_true`); an OPF failure
raises `OPFDispatchError`, both generators skip and record the window, and an invalid dispatch is refused.
Also passing: `tests/test_hif_operating_point_regulation.py`, `tests/test_ieee14_opendss_solve_tolerance.py`,
`tests/test_generate_measurements_hif_physical.py`, `tests/test_waveform_noise_alignment.py` and the
multiscan-estimator unit test of `_scan_operating_points`.

`tests/test_generate_measurements_imbalance_physical.py` asserts that the pypower OPF vector differs from the
unbalanced sensor mean by more than the paired OpenDSS reference does (`opf_gap > gap`). Under the new default
the two references are the same operating point and differ by 2e-4 pu, so the assertion fails by 3e-7. The
test's premise ("different dispatch than the OpenDSS solve") no longer holds; it should assert that
`z_reference_opf` and `z_true` agree (say within 5e-4 pu) or run the generator with `--dispatch-mode case14`.

## Open items

- `research/gnn_screen/dagger_corpus.py` replays unbalance rows with load scaling only; for the 20260923opf
  unbalance corpus its `imbalance_balanced` path must also call `gi._apply_operating_point_dispatch(op_point)`.
- `pipeline.env`, the scenario-generator `PHYSICAL_HIF_*` constants and the GNN converter defaults still name
  the 20260923b corpora.
- At the same OPF operating point the two simulators still differ by about 2e-4 pu in reactive injections
  (the OpenDSS Vsource's MVASC3 = 5e6 short-circuit impedance against pypower's ideal slack) and 9e-6 pu in
  voltage magnitude: 0.02 SCADA sigma, but about 2 sigma at 1e-4 PMU precision. It does not affect the
  estimators, which replay OpenDSS, but PMU data synthesized for the pypower families from pypower solutions
  would carry it as a simulator cue unless the source impedance is matched.
