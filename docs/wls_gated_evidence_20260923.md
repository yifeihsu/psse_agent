# WLS-gated diagnostics: the 2026-09-23 evidence contract

The IEEE-14 DAgger research study (12 families) moves from `scada_only` to
`evidence_profile="wls_gated_diagnostics"`, the new research default in
`psse_env/evidence_profile.py`. The contract decided by the user:

1. No fault flag, seeded sensor signature, family or correction hint, and no
   precomputed diagnosis reaches the agent on any family. Detection uses only the
   balanced SCADA snapshot and the balanced WLS (chi-square at alpha 0.01 OR a
   maximum normalized residual of 4.0).
2. After a current balanced WLS alarm on the active state the agent may request
   the auxiliary streams that the ground truth generated for that root and run
   the matching diagnostics: three-phase PMU phasors on HIF and unbalance roots,
   spectra on harmonic roots, breaker telemetry on topology roots, multi-scan
   windows on HIF and parameter roots only.
3. `run_alternative_test` and any learned (GNN) screen stay blocked.
4. PMU phasors are introduced specifically for HIF. The HIF corpora are to be
   regenerated with three-phase voltage and branch-current phasor noise of
   `1e-4` pu per rectangular component (the 2026-09-19/21/23 corpora carry
   `5e-3` / `1e-3`). The unbalance corpus keeps its sigmas.

This document covers the generation, corpus and pipeline side of the change
(the controller, provider and expert side is documented with the profile in
`psse_env/evidence_profile.py`, `tests/test_wls_gated_boundary.py` and
`psse_env/oracle/test_wls_gated_routing.py`).

## The profile in one table

| profile | seeded flags, hints, diagnoses | auxiliary streams in metadata | tools beyond balanced SCADA | default |
| --- | --- | --- | --- | --- |
| `scada_only` | never | stripped at generation and reset | none (every auxiliary tool refused) | until 2026-09-22 |
| `wls_gated_diagnostics` | never | kept, released only after a current WLS alarm | three-phase / harmonic / NLM / HSE / HIF estimators, gated; `run_alternative_test` and learned screens blocked | from 2026-09-23 |
| `auxiliary_diagnostics` | allowed (historical reproduction) | kept, always available | all | explicit override only |

Both strict profiles (`scada_only`, `wls_gated_diagnostics`) share
`is_strict_boundary`: flagged waveform-signature generation is a `ValueError`,
undeclared historical training rows are rejected by
`validate_training_evidence_profile`, the evaluator inserts and charges the
setup `run_wls` before any historical setup context, and hidden truth,
`release_audit` and the `true_*_errors` stay on the audit side of the release
partition.

## What each family's root carries

Execution metadata of a generated root under `wls_gated_diagnostics`
(`Round0ScenarioGenerator.build` followed by `sanitize_execution_for_profile`;
the same allowlist, `sanitize_gated_metadata`, is applied by the suite
partition). Every root records `metadata.evidence_profile`, the manifest entry
and the generator report record it too.

| family | balanced SCADA | auxiliary streams kept | never present |
| --- | --- | --- | --- |
| no_error, measurement, multi_measurement | `sigma_z` | none | labels, `op_point` |
| parameter, measurement+parameter | `sigma_z` | `parameter_scans` (`z_scans`, `sigma_z`, `scan_indices`) | truth-perturbed `initial_states`, scan `op_point` |
| topology, measurement+topology | `sigma_z`, `operator_noise`, `structural_zero_indices` | `substation_telemetry`, `reported_breaker_status`, `operator_layout`, meter nodes, model id/fingerprint | `true_cb_closed`, the true breaker |
| harmonic | `sigma_z` | `harmonic_measurements`, `harmonic_orders` | `source_bus`, THD target |
| hif, measurement+hif | `sigma_z`, `noise_contract` (scada channel), `measurement_convention` | `three_phase_voltages`, `three_phase_branch_currents`, `three_phase_sigma`, `branch_current_sigma_pu`, `hif_runtime` (current acquisition and its scan `op_point`), `hif_scan_window` (`scans` with `z_obs`, phasors, `op_point`; `scan_window_path` = opaque root id) | `nlm_diagnostic`, `faulted_model_dir`, `z_clean`, `*_clean` phasors, `window_metadata`, label, the corpus row id |
| three_phase_unbalance | `sigma_z`, `noise_contract` | `three_phase_voltages`, `three_phase_branch_currents`, sigmas | `target_bus`, `op_point`, label |
| telemetry_no_disturbance | `sigma_z`, `noise_contract` | balanced phasors at the row's sigma | `telemetry_control_semantics` (dropped by the allowlist), `op_point` |

Generation admission is the WLS anomaly test in every profile (the same
`_require_anomalous` gate as `scada_only`, margin 1.25), so the admitted window
set of a corpus does not depend on the profile. Under `wls_gated_diagnostics`
the waveform builders additionally require the stream the diagnostics need:
HIF rows need scans (`hif_scans_missing`) and phasors
(`three_phase_voltages_missing`), unbalance rows need phasors and a telemetry
observable unbalance (`unbalance_not_observable`, the VUF or current-spread
gate the auxiliary profile applied; the committed detectable unbalance subsets
already passed it), harmonic rows need a spectrum. The cached corpus diagnosis
(`nlm_diagnostic`) is neither required nor carried: strict roots recompute from
the phasors.

## PMU precision and OPF dispatch: the 20260923opf corpora

The regeneration landed on 2026-09-23 (corpus package WP4,
`docs/opf_operating_points_20260923.md`; receipts, replay reports and audits
under `output/hif_physical_revision_20260923opf/`). The corpora carry the tag
`20260923opf`: the six HIF recipes at phasor sigma `1e-4` per rectangular
component (`three_phase_sigma`, `branch_current_sigma_pu` and the
`noise_contract` in `meta.json`, every row and every scan) with OPF-driven
operating points (unit dispatch, PV setpoints and source voltage from the
pypower AC-OPF at each scan's per-bus loads, the law the pypower families use)
on the regulated, tightly converged OpenDSS model, and the 440-window unbalance
corpus at the same OPF-driven operating points with unchanged sigmas
(5e-3 / 1e-3). Seeds and recipes are those of 20260923b: fault labels, load
profiles, SCADA noise draws and healthy-control rows are bit-identical row by
row (777 HIF windows, 7,770 scans, 440 unbalance windows compared); only the
dispatch, hence the clean physics, and the HIF phasor precision changed. No OPF
failed, so no window or control was skipped. The generator constants resolve the
detectable subsets by name (`resolve_tagged_corpus_path`):
`PHYSICAL_HIF_SAMPLE_PATHS` are the four
`hif_physical69_main_{train,valid,train_extra,valid_extra}_detectable_{27,8,77,19}x10_20260923opf`
subsets, `PHYSICAL_IMBALANCE_SAMPLE_PATH` is
`out_measurements_imbalance_currents_ybus_detectable_162_20260923opf`, and the
detection-limit and sweep corpora follow the tag. The committed 20260923b
subsets stay addressable as `PHYSICAL_HIF_SAMPLE_PATHS_20260923B` and
`PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B`.

Alarms and admissions (discovered-mode WLS admission at margin 1.25, reference
scan; healthy controls kept):

| corpus | windows | reference-scan alarms 20260921 -> 20260923/b -> 20260923opf | admitted 20260921 -> 20260923/b -> 20260923opf | phasor sigma V / I, 20260923opf |
| --- | --- | --- | --- | --- |
| `hif_physical69_main_train_84x10` | 84 | 37 -> 39 -> 39 | 25 -> 27 -> 27 (same ids) | 1e-4 / 1e-4 |
| `hif_physical69_main_valid_21x10` | 21 | 10 -> 11 -> 11 | 7 -> 8 -> 8 (same ids) | 1e-4 / 1e-4 |
| `hif_physical69_main_train_extra_252x10` | 252 | 117 -> 126 -> 122 | 69 -> 77 -> 77 (same ids) | 1e-4 / 1e-4 |
| `hif_physical69_main_valid_extra_63x10` | 63 | 28 -> 31 -> 30 | 17 -> 19 -> 19 (same ids) | 1e-4 / 1e-4 |
| `hif_physical69_detection_limit_21x10` | 21 | 0 (20260923opf; no alarm in any generation) | 0 | 1e-4 / 1e-4 |
| `hif_physical_sweep_eval_336x10` | 336 | 70 (20260923opf; 6 paired healthy alarms) | 57 | 1e-4 / 1e-4 |
| `out_measurements_imbalance_currents_ybus_440` | 440 | 218 at margin 1.0 -> 218 -> 218 | 160 -> 160 -> 162 (157 common, 5 new, 3 dropped) | 5e-3 / 1e-3 (unchanged) |

The 20260923 and 20260923b generations differ only in physics (regulated
reactive limits, then OpenDSS tolerance 1e-8); their admissions are identical to
the window (131 HIF, 160 unbalance). The 20260923opf re-audit admits the same
27 + 8 + 77 + 19 = 131 HIF windows with identical admitted ids corpus by corpus:
the phasor sigma does not touch SCADA (the phasor noise is drawn after the SCADA
noise and consumes the same normals at any scale), and the OPF dispatch moved
only four reference-scan alarms in the two extra corpora (122 and 30 against 126
and 31) without changing which windows clear the margin. Healthy-control alarms
are unchanged (1/20, 0/5, 1/60, 0/15; paired healthy alarms 1, 0, 6, 1; sweep
6/336; detection limit none). The unbalance corpus admits 162 of 440 windows
(160 before; 157 in common, 5 newly admitted, 3 no longer; two
`unbalance_not_observable` rejections against three), 218 alarms at margin 1.0
as before, controls 2/60 at margin 1.0 and 0/60 at margin 1.25 as before. The
five detectable subsets re-audit to every window admitted.

Dispatch statistics of the healthy reference (scan 0's operating point, fault
removed; min / median / max over the windows), 20260923b -> 20260923opf:
HIF bus-3 P injection -1.176 / -0.979 / -0.754 pu -> -0.773 / -0.648 / -0.641 pu;
HIF bus-8 Vm 1.081 / 1.090 / 1.090 -> 1.060 / 1.060 / 1.060; HIF bus-8 Q
injection 0.142 / 0.183 / 0.240 -> 0.056 / 0.087 / 0.118; unbalance bus-3 P
-1.177 / -0.954 / -0.754 -> -0.773 / -0.652 / -0.641; unbalance bus-8 Vm
1.081 / 1.090 / 1.090 -> 1.060 / 1.060 / 1.060. An OPF root sample built with
the scenario generator's `_solve_ac_opf` at the very same load scales gives the
20260923opf numbers to the digit (maximum gap 0.0 pu on bus-3 P and bus-8 Vm
over 777 HIF and 440 unbalance windows): the OpenDSS families now sit on the
pypower families' dispatch. What remains between the two simulators at one
operating point is the 2e-4 pu reactive offset of the Vsource's finite
short-circuit impedance (0.02 SCADA sigma; 2 sigma at PMU precision).

Estimator gate at 1e-4: the multi-scan HIF estimator (shared resistance, all
ten scans) was run on the 27 windows of the two valid detectable subsets and
its result put through the provider's unchanged fail-closed acceptance
(`MatpowerDeploymentProviders._hif_diagnostic_acceptance`: weighted residual
norm at most 3.0, residual reduction against the no-HIF null at least 0.20 or
a conclusive terminal-current differential, no model mismatch). 27/27 are
accepted on `residual_reduction_vs_null`: weighted residual norm 0.972-1.026
(median 0.999, the noise floor), no-HIF null norm 108-383 (median about 230),
no model mismatch, no estimator failure, phase correct in all 27, median
|alpha| error 1e-4, about 150 s per window. The 104 training-subset windows
were not run.

The pipeline guards the precision: `research/hpc/full_pipeline_20260907/pipeline.env`
declares `PMU_PHASOR_SIGMA=1e-4` and `prerequisites.sh` refuses, under
`wls_gated_diagnostics`, HIF corpora whose `meta.json` does not declare that
value for both `three_phase_sigma` and `branch_current_sigma_pu` (set
`PMU_PHASOR_SIGMA=` to run an explicit ablation on the 5e-3 corpora); the cell
names the five detectable 20260923opf subsets (`HIF_CORPUS_*`,
`IMBALANCE_CORPUS`) and `research/test_hpc_full_pipeline.py` asserts the names
and the capacities (131 HIF, 162 unbalance; plans unchanged at 78 + 40 HIF
roots, 90 unbalance roots and 40 balanced controls).
`tests/test_wls_gated_generation.py::test_regenerated_hif_corpora_declare_the_pmu_phasor_sigma`
runs the same check on the `20260923opf` corpora (it was skipped while they
were pending). The GNN screen converter (`research/gnn_screen/dagger_corpus.py`)
also defaults to the 20260923opf corpora and replays each unbalance row's stored
dispatch after its uniform load scaling, so its balanced references reproduce.

## Known family cues

Reviewed in the design maps; recorded so results are read with them in mind.

Resolved by the 20260923opf corpora:

- Dispatch provenance of the OpenDSS families. Until 20260923b the HIF,
  measurement+HIF, unbalance and telemetry_no_disturbance roots came from
  OpenDSS at the model's default dispatch (bus 3/6/8 units as condensers,
  bus-8 voltage 1.07-1.10) while the other eight families came from the
  pypower AC-OPF, so the SCADA values themselves separated the two groups
  (bus-3 P -1.18..-0.75 against -0.79..-0.62 pu, bus-8 Vm 1.08..1.10 against
  1.06). The 20260923opf corpora apply the same AC-OPF dispatch to the OpenDSS
  solves; the dispatch statistics above match the OPF root sample exactly,
  and the cell no longer names a 20260923b corpus.

Deliberately not changed (accepted by the user for this study):

- Topology noise model. Topology roots declare structural zeros at P7/Q7 (sigma
  0, exact constraints) and the two-meter sigma 0.0141421 at P3/Q3 through
  `operator_noise` / `structural_zero_indices`; every other family uses the flat
  0.001 / 0.01 profile with noisy P7/Q7. The declaration is execution metadata
  in both strict profiles.
- Availability-based cues. The auxiliary streams exist only where the ground
  truth generated them (phasors on HIF/unbalance/telemetry-control roots,
  spectra on harmonic roots, breaker telemetry on topology roots, repeated
  scans on parameter and HIF roots). After an alarm, which acquisition answers
  is therefore itself informative. The user accepted this as the study's
  contract ("the streams the ground truth generated for that root") rather than
  a uniform PMU/scan window on all 12 families.
- HIF estimators use simulator operating points. `hif_runtime.load_scale` /
  `op_point` and the per-scan `op_point` of `hif_scan_window` are the
  generator's operating points, not values estimated from measurements; the
  single-scan and multi-scan HIF estimators replay OpenDSS at them. They stay in
  the gated metadata because the estimators need them; they never enter the
  policy observation.
- Numeric lattice. Synthesized pypower topology and harmonic vectors are
  quantized to 1e-12; corpus and OpenDSS families keep full binary64 digits.
  Canonicalizing every family at record time would change every corpus root's
  measurement bytes and physical-root fingerprint (protected-suite matching,
  frozen BC0 suite, exactness checks on healthy channels), so it is not a
  one-line change and was left as is.
- Admission shaping. Every fault family must alarm with margin 1.25; topology
  also needs branch dominance and node/breaker rank 1, parameter needs ranking
  dominance. Unchanged.

## Launch procedure

1. Done 2026-09-23 (WP4). The corpora were regenerated with
   `scripts/regenerate_hif_physical_corpora.py --tag 20260923opf` (defaults:
   `--dispatch-mode opf`, HIF `--three-phase-noise-pu 1e-4
   --branch-current-noise-pu 1e-4`, unbalance sigmas unchanged), validated,
   audited, subset and re-audited. The detectable subsets are
   `hif_physical69_main_{train,valid,train_extra,valid_extra}_detectable_{27,8,77,19}x10_20260923opf`
   and `out_measurements_imbalance_currents_ybus_detectable_162_20260923opf`
   (tracked with `git add -f`; the full corpora, detection limit and sweep stay
   local); the generator constants resolve them by name.
2. Done 2026-09-23 (WP6). `pipeline.env` names them (`HIF_CORPUS_TRAIN`,
   `HIF_CORPUS_VALID`, `HIF_CORPUS_TRAIN_EXTRA`, `HIF_CORPUS_VALID_EXTRA`,
   `IMBALANCE_CORPUS`), the capacity comment reads 131 HIF and 162 unbalance
   windows (the plans use 78 + 40 HIF roots and 50 + 2x12 + 16 = 90 unbalance
   roots plus 40 balanced controls), `research/test_hpc_full_pipeline.py`
   asserts the names, and the `TODO(20260923opf)` markers are gone.
   `EVIDENCE_PROFILE` defaults to `wls_gated_diagnostics` and
   `HIF_SIGNATURE_MODE` to `discovered`.
3. Run `research/test_hpc_full_pipeline.py`, `tests/test_wls_gated_generation.py`
   (the sigma test now runs), `tests/test_scada_only_generation.py`,
   `psse_env/dagger/test_research_dagger_minimal.py`,
   `research/gnn_screen/tests`, then `bash -n` on the cell scripts
   (`deploy_remote.sh` does this remotely as well).
4. Deploy with `deploy_remote.sh BUNDLE BRANCH COMMIT PIPE_DIR` into a fresh
   pipeline directory; `prerequisites.sh` checks the corpora exist and declare
   `PMU_PHASOR_SIGMA`. Leave `PREVIOUS_PIPE` empty: the receipt profile check
   (`assert_stage_evidence_profile`) refuses to reuse the D0, suite or BC0 of a
   `scada_only` or auxiliary cell, and undeclared archives count as auxiliary.
5. Submit `submit_pipeline.sh` (chain d0 bc0 r1c r1t r1e r2c r2t r2e). Stage 0
   builds D0 and the suites with `--evidence-profile wls_gated_diagnostics
   --hif-signature-mode discovered` and records the profile in `d0.done`,
   `suite.done`; stage 1 validates every training row's declared profile before
   BC0; collection and evaluation pass the same profile to the runner, which
   records it in `research_profile.evidence_profile` and in the mixture report.
