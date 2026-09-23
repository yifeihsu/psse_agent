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

## PMU precision: regeneration status

The PMU regeneration is pending and owned by the corpus package (WP4). The
final corpora carry the tag `20260923opf`: HIF at phasor sigma `1e-4` with
OPF-driven operating points (dispatch, PV setpoints and source voltage from the
AC-OPF at the window's load scale, as the pypower families already use) on the
regulated, tightly converged OpenDSS model, and the unbalance corpus with the
same OPF-driven operating points and unchanged sigmas. Because the dispatch
changes the SCADA values, the detectable counts are unknown until the
regeneration runs; the generator resolves them from the directory names
(`resolve_tagged_corpus_path`), and `PHYSICAL_HIF_SAMPLE_PATHS`,
`PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH`, `PHYSICAL_HIF_SWEEP_SAMPLE_PATH` and
`PHYSICAL_IMBALANCE_SAMPLE_PATH` follow `PHYSICAL_HIF_CORPUS_TAG =
"20260923opf"`. Until the directories exist they resolve to `PENDING`
placeholders that no source lookup accepts; the committed 20260923b subsets stay
addressable as `PHYSICAL_HIF_SAMPLE_PATHS_20260923B` and
`PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B`.

Alarms and admissions of the committed generations (discovered-mode WLS
admission at margin 1.25, reference scan; healthy controls kept):

| corpus | windows | reference-scan alarms 20260921 -> 20260923/b | admitted 20260921 -> 20260923/b | phasor sigma V / I |
| --- | --- | --- | --- | --- |
| `hif_physical69_main_train_84x10` | 84 | 37 -> 39 | 25 -> 27 | 5e-3 / 1e-3 |
| `hif_physical69_main_valid_21x10` | 21 | 10 -> 11 | 7 -> 8 | 5e-3 / 1e-3 |
| `hif_physical69_main_train_extra_252x10` | 252 | 117 -> 126 | 69 -> 77 | 5e-3 / 1e-3 |
| `hif_physical69_main_valid_extra_63x10` | 63 | 28 -> 31 | 17 -> 19 | 5e-3 / 1e-3 |
| `out_measurements_imbalance_currents_ybus_440` | 440 | 218 (margin 1.0) | 160 | 5e-3 / 1e-3 |
| `*_20260923opf` (pending) | same recipes and seeds | to be audited | to be audited | HIF 1e-4 / 1e-4; unbalance unchanged |

The 20260923 and 20260923b generations differ only in physics (regulated
reactive limits, then OpenDSS tolerance 1e-8); their admissions are identical to
the window (131 HIF, 160 unbalance). Changing only the phasor sigma leaves the
SCADA draws unchanged (the phasor noise is drawn after the SCADA noise from the
same generator and consumes the same number of normals at any scale), so a
sigma-only regeneration would keep 27 + 8 + 77 + 19; the OPF dispatch does
change SCADA, so the `20260923opf` admissions must be re-audited.

The pipeline guards the precision: `research/hpc/full_pipeline_20260907/pipeline.env`
declares `PMU_PHASOR_SIGMA=1e-4` and `prerequisites.sh` refuses, under
`wls_gated_diagnostics`, HIF corpora whose `meta.json` does not declare that
value for both `three_phase_sigma` and `branch_current_sigma_pu` (set
`PMU_PHASOR_SIGMA=` to run an explicit ablation on the 5e-3 corpora).
`tests/test_wls_gated_generation.py::test_regenerated_hif_corpora_declare_the_pmu_phasor_sigma`
runs the same check on the `20260923opf` corpora once they exist and is skipped
while they are pending.

## Known family cues deliberately not changed

These were reviewed in the design maps and accepted by the user for this
study; they are recorded so results are read with them in mind.

- Dispatch provenance of the OpenDSS families. HIF, measurement+HIF, unbalance
  and telemetry_no_disturbance roots come from OpenDSS at the model's default
  dispatch (bus 3/6/8 units as condensers, bus-8 voltage 1.07-1.10), the other
  eight families from the pypower AC-OPF; the SCADA values themselves separate
  the two groups. The `20260923opf` regeneration adopts the OPF dispatch for the
  OpenDSS corpora and removes this cue; the 20260923b corpora still carry it.
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

1. Regenerate the corpora (WP4): the six HIF recipes with
   `--three-phase-noise-pu 1e-4 --branch-current-noise-pu 1e-4` and the OPF
   reference dispatch, the 440-window unbalance corpus with the OPF reference
   dispatch, tag `20260923opf`; validate, audit, subset, re-audit. The detectable
   subsets appear as `hif_physical69_main_{train,valid,train_extra,valid_extra}_detectable_<N>x10_20260923opf`
   and `out_measurements_imbalance_currents_ybus_detectable_<N>_20260923opf`;
   the generator constants resolve them by name.
2. Point `pipeline.env` at them: `HIF_CORPUS_TRAIN`, `HIF_CORPUS_VALID`,
   `HIF_CORPUS_TRAIN_EXTRA`, `HIF_CORPUS_VALID_EXTRA`, `IMBALANCE_CORPUS`, the
   capacity comment (the plans use 78 + 40 HIF roots and 50 + 2x12 + 16 = 90
   unbalance roots plus the balanced control), and the corpus names in
   `research/test_hpc_full_pipeline.py`; remove the two `TODO(20260923opf)`
   markers. `EVIDENCE_PROFILE` already defaults to `wls_gated_diagnostics` and
   `HIF_SIGNATURE_MODE` to `discovered`.
3. Run `research/test_hpc_full_pipeline.py`, `tests/test_wls_gated_generation.py`
   (the pending sigma test now runs), `tests/test_scada_only_generation.py`,
   `psse_env/dagger/test_research_dagger_minimal.py`, then `bash -n` on the cell
   scripts (`deploy_remote.sh` does this remotely as well).
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
