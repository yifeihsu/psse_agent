# Revision plan: IEEE-14 HIF parameters in physical ohms on local voltage bases

Date: 2026-09-19. Original hand-off status: implementation interrupted mid-way (session rate limits).

Implementation and corpus regeneration completed on 2026-09-20; see
[the executed revision and results](ieee14_hif_legacy_reconfiguration_20260919.md).
All physical replays passed. Four sweep top-1 ranking misses and the discovered-mode
HPC capacity shortfall are retained as explicit experimental findings below that report.
The remainder of this document preserves the original hand-off plan. This document is the
hand-off for the session that finishes the work. It is self-contained: the design, what already landed on
disk, what remains per file, the verification commands, and the corpus regeneration steps.

How to use it in a fresh session: work the steps in order, one step at a time, running the listed tests
after each. Do not use multi-agent orchestration; the remaining work is sequential and mostly mechanical.
Never modify anything under `artifacts/` (tracked corpora) or `IEEE_14_OpenDSS/*.dss`. Research code:
when a digest or registry hash pin blocks a legitimate change, remove the pin and say so.

## 0. Design (fixed; do not re-litigate)

Goal. The legacy IEEE-14 HIF stack (normalized `BasekV=1` OpenDSS model, `three_phase_nlm`,
`Transmission/generate_measurements_hif_ieee14.py`, the estimators, the agent tool surface) specified HIF
magnitude as system per-unit with no voltage meaning. It now specifies fault resistance in **physical ohms
at the faulted line's actual voltage** (buses 1-5: 69 kV, 6-7 and 9-14: 13.8 kV, bus 8: 18 kV, 100 MVA)
and converts per line:

    R_pu        = R_ohm / (kV_local^2 / 100)        local base; 69 kV: Zbase 47.61 ohm, 13.8 kV: 1.9044, 18 kV: 3.24
    R_model_ohm = R_pu * 0.01                       what the normalized 1 kV DSS model needs

The DSS files are not edited (a consistently based pu solution is base-invariant). Physical kV must never
reach `dss_hif_injector`, `constant_impedance_hif_kw`, `hif_ohms_from_pu` or the legacy NLM bridge.

Populations. Main training: 100-1000 ohm on 69 kV lines (2.10-21.00 pu), bands 100-200 / 200-500 /
500-1000, weights early 2:2:1, full 1:1:1. Detection-limit: 1000-5000 ohm on 69 kV (21-105 pu).
Evaluation sweep {50,100,200,500,1000,2000,5000} ohm at both 69 and 13.8 kV. Sixteen eligible lines;
`Line.7-8` (13.8/18 kV) excluded from generation but still accepted by estimators on its from-bus base.
1000 pu at 69 kV is 47.6 kOhm and is not an HIF.

Classification (`three_phase_model.voltage_bases.hif_resistance_class`, physical ohms, lower bound
inclusive): <50 low_resistance_fault; 50-100 moderately_resistive; 100-200 moderately_high_resistance;
200-500 representative_hif; 500-1000 weak_hif; 1000-5000 extreme_weak_hif; >=5000 near_open_circuit.

Label contract. New corpora: `resistance_units="ohm_local_base"`, `r_hif_ohm` = PHYSICAL ohms,
`r_hif_model_ohm` = model ohms (r_pu*0.01), `r_hif_pu` = local-base pu, plus `local_kv_ll`, `local_kv_ln`,
`zbase_ohm`, `kv_ln` (kept = model 0.577 with `kv_ln_semantics`), `resistance_class`,
`expected_detectability`, `voltage_stratum`, `voltage_base_profile`, nominal current/power. Legacy corpora
(no marker): `r_hif_ohm = r_pu*0.01` model ohms, `kv_ln = 0.577`. Read labels only through
`three_phase_nlm.hif_units.label_model_ohm / label_physical_ohm / label_local_kv_ll`.

Shunt convention. `IEEE_14_OpenDSS/measurement_convention.py`: `"legacy_injection"` (exporter added the
19 Mvar bus-9 capacitor to Qinj) vs `"ybus"` (MATPOWER makeSbus; matches the operator WLS which keeps Bs in
Ybus). The legacy convention made every OpenDSS HIF window alarm the WLS at bus 9 (J~250) regardless of
resistance. Regenerated corpora declare `measurement_convention` on rows and scans; estimators simulate
candidates under the observation's convention; absent marker = legacy (tracked corpora replay unchanged).

Tool surface. Both HIF estimator tools gain `r_hif_ohm_min` (default 50.0) and `r_hif_ohm_max`
(default 5000.0), physical ohms on the candidate line's local base; `r_hif_pu_min/max` become optional
nullable overrides with no default; supplying both units is an error. Provider
`MatpowerDeploymentProviders(hif_resistance_search="physical_ohm"|"legacy_pu")`; `legacy_pu` reproduces the
old 5-1000 system-pu box for replaying old runs.

### 0.1 Shared API already on disk (import; never redefine)

`three_phase_nlm/hif_units.py`: constants `S_BASE_MVA`, `MODEL_KV_LL=1.0`, `MODEL_ZBASE_OHM=0.01`,
`VOLTAGE_BASE_PROFILE_ID`, `RESISTANCE_UNITS_OHM_LOCAL_BASE`, `RESISTANCE_UNITS_PU_LEGACY`,
`DEFAULT_HIF_SEARCH_OHM=(50,5000)`, `LEGACY_HIF_SEARCH_PU=(5,1000)`, `MAIN_HIF_BAND_OHM`,
`DETECTION_LIMIT_BAND_OHM`, `EVALUATION_SWEEP_OHM`, `VOLTAGE_STRATA`, `PHYSICAL_ELIGIBLE_HIF_BRANCHES`
(rows 0-6, 10-12, 14-19), `EXCLUDED_CROSS_VOLTAGE_BRANCHES` (row 13). Functions `line_endpoint_kv`,
`line_kv_ll_for_row0`, `resolve_line_kv_ll(row0, kv_ll=None)`, `eligible_rows_for_stratum`,
`voltage_stratum_for_kv`, `local_pu_from_ohm`, `ohm_from_local_pu`, `model_ohm_from_local_pu`,
`model_ohm_from_physical_ohm`, `physical_ohm_from_model_ohm`,
`hif_resistance_record(*, branch_row0, resistance_ohm=None | r_hif_pu=None, kv_ll=None)`,
`label_resistance_units`, `label_model_ohm`, `label_physical_ohm`, `label_local_kv_ll`,
`resolve_resistance_search_box(*, branch_row0, r_hif_pu_min=None, r_hif_pu_max=None, r_hif_ohm_min=None,
r_hif_ohm_max=None, kv_ll=None, default_ohm=DEFAULT_HIF_SEARCH_OHM, default_pu=None)`.

`IEEE_14_OpenDSS/measurement_convention.py`: `SHUNT_CONVENTION_LEGACY`, `SHUNT_CONVENTION_YBUS`,
`MEASUREMENT_CONVENTION_KEY`, `validate_shunt_convention`, `measurement_convention_payload`,
`resolve_shunt_convention(explicit=None, *source_mappings)`.

`three_phase_model/voltage_bases.py`: `HIF_RESISTANCE_CLASSES_OHM`, `HIF_DETECTION_LIMIT_BAND_OHM`,
`hif_resistance_class`, `hif_resistance_classification_table` (added 2026-09-19 next to the existing
`IEEE14_NOMINAL_KV`, `impedance_base_ohm`, `hif_resistance_spec`, `ieee14_hif_branch_eligibility`).

`hif_search_limits.validate_hif_resistance_box(*, r_hif_pu_min, r_hif_pu_max, r_hif_ohm_min, r_hif_ohm_max)`.

### 0.2 Agreed signatures (implement exactly where still missing)

1. `extract_measurement_series(*, buses=None, branch_names=None, branch_element_overrides=None, shunt_convention="legacy_injection")`.
2. `_simulate_base(..., shunt_convention="legacy_injection")`, `_simulate_candidate(..., shunt_convention="legacy_injection")`, `simulate_hif_candidate(..., shunt_convention=None)`; `_simulate_candidate` output keeps `r_hif_ohm` (model ohms) and adds `r_hif_model_ohm`, `shunt_convention`.
3. `estimate_hif_location_magnitude(..., r_hif_pu_min=None, r_hif_pu_max=None, r_hif_ohm_min=None, r_hif_ohm_max=None, kv_ll=None, shunt_convention=None, resistance_search="physical_ohm")` and the multiscan twin. Payload: `estimated.r_hif_ohm` PHYSICAL; add `r_hif_model_ohm`, `local_kv_ll`, `impedance_base_ohm`, `resistance_basis`, `resistance_class`, `voltage_base_profile`; `i_hif_amp` and `g_hif_siemens` physical (`v_phys = fault_v_volts * kv_ll`); `p_hif_kw` unchanged; multiscan `r_hif_ohm_range`/`i_hif_amp_range` physical; `search` block echoes `r_hif_pu_min/max`, `r_hif_ohm_min/max`, `kv_ll`, `impedance_base_ohm`, `resistance_basis`, `voltage_base_profile`, `box_source`, `shunt_convention`, `resistance_search`.
4. Tool schema and provider as in section 0. Providers read only explicitly supplied arguments (`arguments.get`) and forward them; the `_logic` functions in `mcp_server/matpower_server.py` forward to the estimators.
5. `measurement_convention` on corpus rows/scans/meta; scenario generator allowlists it and copies the row payload into `metadata.hif_runtime`, `metadata.hif_scan_window` and `metadata["measurement_convention"]`.
6. Generator CLI: `--resistance-units {ohm,pu}` (default ohm), `--r-hif-ohm-min/--r-hif-ohm-max`, `--r-hif-ohm-bands "100:200,200:500,500:1000"` + `--r-hif-ohm-band-weights "1,1,1"`, `--r-hif-ohm-sweep "50,100,..."`, `--voltage-stratum {69kv,13p8kv,all_same_voltage}` (default 69kv), `--shunt-convention` (default ybus in ohm mode, legacy_injection in pu mode); `--r-hif-pu-min/max` only with `--resistance-units pu`.
7. Scenario generator constants: `PHYSICAL_HIF_SAMPLE_PATHS` (main train + valid), `PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH`, `PHYSICAL_HIF_SWEEP_SAMPLE_PATH`; `CURRENT_TELEMETRY_HIF_SAMPLE_PATHS` unchanged.

## 1. State on disk at hand-off (probed 2026-09-19, all modules import cleanly)

| Area | File | State |
| --- | --- | --- |
| Shared | `three_phase_nlm/hif_units.py`, `IEEE_14_OpenDSS/measurement_convention.py`, classification in `three_phase_model/voltage_bases.py` | Done and sanity-checked (16 eligible rows; 500 ohm on Line.2-3 = 10.502 pu, class weak_hif; box defaults; legacy label inference) |
| A | `IEEE_14_OpenDSS/export_measurement_series.py` | `shunt_convention` landed (verify the ybus branch skips only the Capacitors block) |
| A | `three_phase_nlm/dss_hif_injector.py` | Landed: model-kV guard, `resistance_ohm`/`kv_ll_local` path, extended `HIFInjectionResult` (verify `to_dict`) |
| A | `three_phase_nlm/branch_current_analysis.py` | Landed: `resolve_line_kv_ll`, physical `r_hif_ohm`, `r_hif_model_ohm`, `cross_voltage_branch` (verify both localization functions pass `kv_ll`) |
| A | `three_phase_nlm/__init__.py` | Exports landed |
| A | `test_export_measurement_series.py` | NOT updated: lines 231-255 still assert Line.7-8 `r_hif_ohm == 1.0` (now 190.44 ohm physical on the 13.8 kV from-bus base) |
| A | `tests/test_hif_units.py`, `tests/test_measurement_convention.py` | ABSENT |
| B | `three_phase_nlm/hif_parameter_estimator.py` | Substantially landed (new kwargs, box, convention, payload). Verify payload fields and `search` block against 0.2.3 |
| B | `three_phase_nlm/hif_multiscan_estimator.py` | PARTIAL: only the convention imports and `HIFScan.shunt_convention` field landed. Signature, `_parse_scans`, box, payload, cache keys not done |
| B | `three_phase_nlm/hif_conditioned_recovery.py`, `conditioned_meter_recovery.py` | Not started (kwarg pass-through) |
| B | `hif_search_limits.py` | Done (`validate_hif_resistance_box`) |
| B | `test_hif_multiscan_estimator.py`, `test_hif_conditioned_recovery.py`, `tests/test_hif_recovery_experiment.py` | Not updated |
| C | `Transmission/generate_measurements_hif_ieee14.py` | Substantially landed (+821 lines; all CLI flags present). Not yet exercised end to end |
| C | `scripts/validate_hif_samples.py`, `validate_hif_multiscan_dataset.py`, `validate_hif_parameter_estimates.py`, `validate_hif_multiscan_parameter_estimates.py`, `validate_branch_current_localization.py`, `build_hif_recovery_stress.py`, `evaluate_hif_measurement_recovery.py`, `benchmark_hif_multiscan_conditions.py` | Not started |
| C | `tests/test_generate_measurements_hif_physical.py` | ABSENT |
| D | `trace_protocol.py` | PARTIAL: constants `HIF_R_OHM_MIN_DEFAULT/MAX_DEFAULT` and the two schema entries landed. Check pu args nullable with no default, glossary text, `summarize_hif_parameter_estimate_payload` passthrough |
| D | `mcp_server/matpower_server.py`, `psse_env/providers/matpower.py`, `psse_env/providers/scenario_generator.py`, `psse_env/dagger/dataset_builder.py`, `psse_env/dagger/evaluator.py`, `psse_env/dagger/release_factories.py`, `schema/sft_trace_decision_schema.json`, `Transmission/build_sft_traces.py`, `eval_sft_agent_gemma_v4.py`, `eval_sft_agent_hardened.py`, `interactive_agent_eval.py` | Not started (no new markers) |
| D | `psse_env/dagger/preliminary_tool_gate.py` | Not changed: the registry-hash `raise` at ~line 305 is still present |
| E | `psse_env/fault_profiles.py`, `research/gnn_screen/practical_corpus.py`, `three_phase_model/disturbances.py` | Landed (detection_limit band, classification, default-resistance rule, receipt class) |
| E | `tests/test_fault_profiles.py`, `tests/test_physical_hif_profile_integration.py`, `tests/test_hif_physical_voltage.py`, `three_phase_model/test_disturbances.py`, `research/gnn_screen/tests/test_practical_corpus.py`, `research/gnn_screen/README.md`, `docs/ieee14_physical_hif_20260918.md` | Modified 16:47-16:49; test outcome not recorded. Run them first |
| F | `research/hpc/full_pipeline_20260907/pipeline.env`, `research/test_hpc_full_pipeline.py` | Landed (corpus paths, 105 capacity, flags) |
| F | `build_suite.py`, `stage_d0.sbatch` (HPC), `docs/ieee14_hif_legacy_reconfiguration_20260919.md`, doc footnotes, `README.md`, `IEEE_14_OpenDSS/AGENTS.md` | Not started (doc ABSENT) |

Baselines recorded before the work: `tests/test_fault_profiles.py tests/test_voltage_bases.py tests/test_hif_physical_voltage.py` 165 passed; `test_hif_search_limits.py psse_env/dagger/test_error_injectors.py` 13 passed; Team E reported 243 passed on its five files before editing. Legacy generator smoke: 2 windows x 3 scans in 8 s.

## 2. Remaining work, in order

Each step names the files, the exact change, and the acceptance test. Steps 1-6 are independent of each
other except where noted, but run them one at a time to keep the session cheap.

### Step 1. Finish Team A (tests only)
- `test_export_measurement_series.py` lines 145, 231, 255: the fixtures inject `r_hif_ohm=1.0` model ohms on `Line.7-8` (row 13, 13.8/18 kV). Keep the injection; change the assertion at line 255 to `estimate["r_hif_ohm"] ~= 100 * 1.9044 = 190.44` and add `estimate["r_hif_model_ohm"] ~= 1.0`, `estimate["cross_voltage_branch"] is True`, `estimate["r_hif_pu"] ~= 100` (already at line 254).
- Add a test: `extract_measurement_series(shunt_convention="ybus")` differs from the legacy series only at index `2*14+8` (Qinj bus 9) by about `0.19 * V9^2` pu (sign per the code's makeSbus convention) and is identical elsewhere. Add a test that `hif_ohms_from_pu(10, kv_ll=69)` raises.
- Create `tests/test_hif_units.py`: 16 eligible rows and the row-13 exclusion record; 7 rows at 69 kV and 9 at 13.8 kV; `hif_resistance_record(branch_row0=2, resistance_ohm=500)` gives `r_hif_pu` 10.502, `r_hif_model_ohm` 0.10502, class `weak_hif`, current 79.67 A; 500 ohm on row 10 gives 262.55 pu; legacy label `{r_hif_pu:124.17, r_hif_ohm:1.2417, kv_ln:0.577, branch_row0:11}` gives `label_model_ohm` 1.2417 and `label_physical_ohm` 124.17*1.9044; new-style label round-trips; `resolve_resistance_search_box` default (1.0502-105.02 pu on row 0), explicit pu, explicit ohm, mixed units raise, `default_ohm=None, default_pu=(5,1000)` legacy; classification at 49, 50, 100, 200, 500, 1000, 5000.
- Create `tests/test_measurement_convention.py`: payload fields, `resolve_shunt_convention` precedence (explicit > scan payload > bare string > legacy), invalid value raises.
- Run: `python -m pytest tests/test_hif_units.py tests/test_measurement_convention.py test_export_measurement_series.py test_branch_current_analysis.py tests/test_voltage_bases.py -q -p no:cacheprovider`

### Step 2. Finish Team B (multiscan estimator, recovery pass-through, tests)
- `three_phase_nlm/hif_multiscan_estimator.py`: mirror the single-scan changes. Signature per 0.2.3. `_parse_scans` reads `measurement_convention` from each scan, falling back to the window, into `HIFScan.shunt_convention`; all scans must agree (ValueError). Call `hif_search_limits.validate_hif_resistance_box` then `hif_units.resolve_resistance_search_box` (legacy_pu -> `default_ohm=None, default_pu=LEGACY_HIF_SEARCH_PU`). Every simulation cache key and `_simulate_candidate_task` kwargs include `shunt_convention`. Payload: `estimated.r_hif_ohm` physical (`median r_pu * impedance_base_ohm`), `r_hif_ohm_range` and `i_hif_amp_range` physical, add `r_hif_model_ohm`, `local_kv_ll`, `impedance_base_ohm`, `resistance_basis`, `resistance_class`, `voltage_base_profile`; `per_scan_r_hif_pu` unchanged; `search` block per 0.2.3. Do not refuse row 13.
- `hif_conditioned_recovery.py`, `conditioned_meter_recovery.py`: forward the new kwargs where they call the estimators (defaults unchanged).
- Verify in `hif_parameter_estimator.py`: `_simulate_candidate` output has `r_hif_model_ohm` and `shunt_convention`; `terminal_current_branch_evidence` passes `kv_ll`; the `estimated` and `search` blocks carry every field in 0.2.3.
- Tests (`test_hif_multiscan_estimator.py`, `test_hif_conditioned_recovery.py`): replace hard-coded 5.0/1000.0 and normalized-ohm assertions; add: default box on a 69 kV line = 1.0502-105.02 pu and on a 13.8 kV line = 26.26-2625.5 pu; `resistance_search="legacy_pu"` reproduces 5-1000 pu; `estimated.r_hif_ohm == r_hif_pu*47.61` on Line.2-3; the `search` payload echoes the box; mixed units raise; convention resolved from scans; `p_hif_kw` equals the old formula.
- Run: `python -m pytest test_hif_search_limits.py test_hif_multiscan_estimator.py test_hif_conditioned_recovery.py tests/test_hif_recovery_experiment.py -q -p no:cacheprovider` (OpenDSS-heavy; minutes).

### Step 3. Finish Team D (tool surface and plumbing)
- `trace_protocol.py`: confirm `r_hif_pu_min/max` are `{"type": ["number","null"], "default": null, "description": ...}` in both tools; glossary (`DECISION_SCHEMA_TEXT`) wording for `r_hif_pu` (local-base pu) and `r_hif_ohm` (physical ohms on that base) plus `local_kv_ll`, `impedance_base_ohm`, `resistance_basis`, `resistance_class`; `summarize_hif_parameter_estimate_payload` passes through `estimated.{r_hif_model_ohm, local_kv_ll, impedance_base_ohm, resistance_basis, resistance_class, voltage_base_profile, r_hif_ohm_range}` and `search.{r_hif_pu_min, r_hif_pu_max, r_hif_ohm_min, r_hif_ohm_max, box_source, shunt_convention}`.
- `mcp_server/matpower_server.py`: `_estimate_hif_location_magnitude_logic` and `_..._multiscan_logic` and both `@mcp.tool` wrappers gain `r_hif_ohm_min`, `r_hif_ohm_max`, `kv_ll`, `shunt_convention`, `resistance_search` (defaults None / "physical_ohm"); `r_hif_pu_min/max` default None; forward all. Document that `_run_three_phase_nlm_logic(r_hif_ohm=...)` takes MODEL ohms.
- `psse_env/providers/matpower.py`: constructor kwarg `hif_resistance_search="physical_ohm"` (validate against {physical_ohm, legacy_pu}); `estimate_hif`/`estimate_hif_multiscan` read only explicit `arguments.get(...)`, pass `resistance_search`, resolve `shunt_convention` from `metadata.hif_runtime` / `hif_scan_window` via `resolve_shunt_convention` (None when undeclared); `hif_summary` gains `local_kv_ll`, `impedance_base_ohm`, `resistance_basis`, `resistance_class`, `r_hif_model_ohm` and the searched box. `psse_env/dagger/release_factories.py`: plumb `hif_resistance_search` where providers are built.
- `psse_env/providers/scenario_generator.py`: allowlist `"measurement_convention"` in `_observable_waveform_scan` (line ~823); in `_hif_scenario` copy the row payload into `metadata.hif_runtime`, `metadata.hif_scan_window`, `metadata["measurement_convention"]`; add the three `PHYSICAL_*` constants (paths in section 4) and export them; update the `_hif_rows` comment (16 lines). `psse_env/dagger/evaluator.py` metadata whitelist (~line 416): add `"measurement_convention"`. `suite_builder._execution_metadata`: confirm nothing strips it.
- `Transmission/build_sft_traces.py`: `call_tool_json` forwards `r_hif_ohm_min/max` (None when absent) and pu bounds only when present; `make_mock_hif_parameter_estimate_payload` uses `label_physical_ohm`/`label_local_kv_ll`; `make_hif_context_payload` sets `r_hif_ohm` from `label_model_ohm(label)` (legacy bridge needs model ohms) and adds `r_hif_ohm_physical`, `resistance_units`. `eval_sft_agent_gemma_v4.py` keep_keys and `eval_sft_agent_hardened.py` DEFAULT_POWER_TOOLS: mirror the schema. `interactive_agent_eval.py`: same hydration semantics if it passes `r_hif_ohm`. `psse_env/dagger/dataset_builder.py` TOOL_JSON_SCHEMAS: add `r_hif_ohm_min/max` `{"type": ["number","null"]}`, make pu ones nullable. `schema/sft_trace_decision_schema.json`: add nullable `local_kv_ll`, `impedance_base_ohm`, `resistance_basis`, `resistance_class`, `r_hif_model_ohm`, `voltage_base_profile`, `r_hif_ohm_range` under `evidence.hif_parameter_estimate` (object is `additionalProperties: true`, so this is documentation-grade).
- `psse_env/dagger/preliminary_tool_gate.py` ~line 305: replace the registry-hash `raise` with a recorded field (`tool_registry_matches_runtime`) and continue; update its test.
- Tests: fix every test pinning the registry hash, the 5.0/1000.0 defaults or the schema text; add provider tests with mocked `_logic` proving no-args -> physical_ohm with no pu bounds, explicit pu forwarded, explicit ohm forwarded, legacy_pu mode, convention resolved from metadata.
- Run: `python -m pytest test_trace_protocol.py psse_env/dagger psse_env/providers/test_scenario_generator.py psse_env/providers/test_matpower_providers.py tests/test_covariance_tool_forwarding.py -q -p no:cacheprovider` (split if slow).

### Step 4. Finish Team C (validators, generator smoke, test)
- Smoke the landed generator first: `python Transmission/generate_measurements_hif_ieee14.py --out <tmp> --n-hif 2 --n-no-error 1 --seed 7 --scans-per-window 2 --resistance-units ohm --r-hif-ohm-sweep 500 --voltage-stratum 69kv` then inspect one row: `r_hif_pu == 500/47.61`, `r_hif_ohm == 500`, `r_hif_model_ohm == r_hif_pu*0.01`, `resistance_units == "ohm_local_base"`, `local_kv_ll == 69`, `resistance_class == "weak_hif"`, `expected_detectability == "weak"`, every scan has `measurement_convention.shunt_convention == "ybus"`, `three_phase_voltages` `kvbase_ln` is 69/sqrt3 for b1 and 13.8/sqrt3 for b6, meta lists eligible rows 0-6 and excludes Line.7-8, and `z_clean[28+8]` agrees with `z_true[28+8]` within 0.03 pu (legacy mode differs by about 0.19). Also confirm the injector received model ohms (`r_hif_model_ohm`) and the legacy NLM diagnostic still runs (`nlm_diagnostic.success`). Fix whatever the smoke reveals.
- Validators: `validate_hif_samples.py` gains `--meta PATH` (expected eligible set from `meta.hif.eligible_branch_row0`, fallback module constant), per-stratum/per-class counts, `--allow-non-top3-detectability CLASSES` (e.g. `weak,extreme`); `validate_hif_multiscan_dataset.py` replays with `resolve_shunt_convention(None, row, meta)` passed to `_simulate_candidate` and accepts a 16-row eligible set from meta; `validate_hif_parameter_estimates.py`, `validate_hif_multiscan_parameter_estimates.py`, `evaluate_hif_measurement_recovery.py` gain `--r-hif-ohm-min/--r-hif-ohm-max` (pu flags kept, default None) and compute truth power from `label_physical_ohm` + `label_local_kv_ll` when `resistance_units` is `ohm_local_base`; `validate_branch_current_localization.py` compares `r_hif_pu` and reports ohms through `hif_units`; `build_hif_recovery_stress.py` writes `resistance_units="pu_legacy_normalized_model"` and keeps model ohms; `benchmark_hif_multiscan_conditions.py` LABEL_KEYS include the new keys if it enumerates them.
- Create `tests/test_generate_measurements_hif_physical.py` from the smoke assertions above, plus the legacy-mode assertion (`--resistance-units pu`, 20-200 pu still writes `r_hif_ohm == r_pu*0.01`, `kv_ln == 0.577`, `resistance_units == "pu_legacy_normalized_model"`).
- Run: `python -m pytest tests/test_generate_measurements_hif_physical.py -q -p no:cacheprovider` and each validator with `--help`.

### Step 5. Verify Team E
- Run: `python -m pytest tests/test_fault_profiles.py tests/test_physical_hif_profile_integration.py tests/test_hif_physical_voltage.py three_phase_model/test_disturbances.py research/gnn_screen/tests/test_practical_corpus.py research/gnn_screen/tests/test_practical_parallel.py -q -p no:cacheprovider`. Fix failures in Team E files only. Confirm `docs/ieee14_physical_hif_20260918.md` has the appended "2026-09-19 additions" section (classification table, detection-limit cohort, default-resistance rule) and `research/gnn_screen/README.md` mentions the detection-limit cohort.

### Step 6. Finish Team F (HPC flags and documentation)
- `research/hpc/full_pipeline_20260907/build_suite.py`: argparse `--hif-sample-paths` (nargs="+", type=Path) and `--imbalance-sample-path` forwarded to `research.resolve_scenario_sources(...)` (signature in `scripts/run_dagger_research.py:151`). `stage_d0.sbatch`: pass `--hif-sample-paths "$HIF_CORPUS_TRAIN" "$HIF_CORPUS_VALID"`. Confirm `pipeline.env` `collection_args()` passes the same flags and its comment names the ohm bands and profile id; README dated paragraph.
- Write `docs/ieee14_hif_legacy_reconfiguration_20260919.md`: decision and scope with the two formulas; voltage map table (buses, kV_LL, Zbase, Ibase, phase voltage) and the note that the 132/33/11 kV annotations in the DSS comment blocks are inert; conversion tables (sweep ohms -> approximate current at 69 kV and pu at 69/18/13.8 kV; legacy 20-200 system pu = 952-9522 ohm at 69 kV and 38-381 ohm at 13.8 kV; legacy box 5-1000 pu = 238 ohm to 47.6 kOhm at 69 kV); classification table (`python -c "from three_phase_model.voltage_bases import hif_resistance_classification_table as t; import json; print(json.dumps(t(), indent=1))"`); populations; label and payload contract; measurement convention; regenerated corpora with the exact commands (section 4); a Results section with the literal line `RESULTS_PENDING: filled after corpus generation and validation`; what remains legacy (tracked 20260903/20260714 corpora and the frozen BC0 suite keep system-pu labels; IEEE 57 has no declared kV map). Plain prose, tables, no em-dashes.
- Footnotes (units note + cross-reference): `docs/gnn_practical_scenarios_20260916.md` (lines 33, 44-51, 81, 110), `docs/branch_current_telemetry_20260903.md` (header note), `docs/fault_scenario_review_20260917.md` (lines 20-21, 113-116), `docs/hif_multiscan_estimation.md` (generation commands 66-74, 133-157), `docs/hif_measurement_recovery_20260916.md` (107-108), `docs/ieee57_disturbance_testing_20260911.md` (one line: pu-only, no kV map). `README.md` docs index: add the 20260918 and 20260919 docs. `IEEE_14_OpenDSS/AGENTS.md`: "Voltage bases (2026-09-19)" section.
- Run: `python -m pytest research/test_hpc_full_pipeline.py -q -p no:cacheprovider`.

### Step 7. Integration run
```bash
python -m pytest tests/test_hif_units.py tests/test_measurement_convention.py test_export_measurement_series.py test_branch_current_analysis.py test_hif_search_limits.py test_trace_protocol.py tests/test_fault_profiles.py tests/test_voltage_bases.py tests/test_hif_physical_voltage.py research/test_hpc_full_pipeline.py psse_env/dagger/test_error_injectors.py psse_env/dagger/test_splits.py psse_env/dagger/test_suite_builder.py psse_env/providers/test_scenario_generator.py -q -p no:cacheprovider
```
Then separately (OpenDSS-heavy): `test_hif_multiscan_estimator.py`, `tests/test_generate_measurements_hif_physical.py`, `tests/test_physical_hif_profile_integration.py`, `test_hif_conditioned_recovery.py`, `tests/test_hif_recovery_experiment.py`, `psse_env/dagger` (whole), `psse_env/providers/test_matpower_providers.py`. Do not weaken tests to hide defects; a failure that predates this work should be reported as pre-existing (reason from `git diff -- <file>`; the tree has other uncommitted work, so do not stash).

## 3. Corpus regeneration (after Step 7 is green)

Run from the repo root (each about 4 s per window plus the legacy NLM; the sweep takes roughly 25 min):
```bash
G=Transmission/generate_measurements_hif_ieee14.py; A=artifacts/measurements
python $G --out $A/hif_physical69_main_train_84x10_20260919 --n-hif 84 --n-no-error 20 --seed 20260919 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands "100:200,200:500,500:1000" --r-hif-ohm-band-weights "1,1,1" --voltage-stratum 69kv
python $G --out $A/hif_physical69_main_valid_21x10_20260919 --n-hif 21 --n-no-error 5 --seed 20260920 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands "100:200,200:500,500:1000" --r-hif-ohm-band-weights "1,1,1" --voltage-stratum 69kv
python $G --out $A/hif_physical69_detection_limit_21x10_20260919 --n-hif 21 --n-no-error 0 --seed 20260921 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-min 1000 --r-hif-ohm-max 5000 --voltage-stratum 69kv
python $G --out $A/hif_physical_sweep_eval_336x10_20260919 --n-hif 336 --n-no-error 0 --seed 20260922 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-sweep "50,100,200,500,1000,2000,5000" --voltage-stratum all_same_voltage
```
For each corpus: `python scripts/validate_hif_samples.py <dir>/samples.jsonl --meta <dir>/meta.json --allow-non-top3-detectability weak,extreme`, `python scripts/validate_hif_multiscan_dataset.py <dir>/samples.jsonl --meta <dir>/meta.json --strict-physics --output <dir>/quality_report.json`, `python scripts/validate_branch_current_localization.py <dir>/samples.jsonl --output <dir>/branch_current_localization_report.json`. The HPC env already points `HIF_CORPUS_TRAIN/VALID` at the first two directories; `prerequisites.sh` refuses uncommitted inputs, so commit the corpora before a cell.

## 4. Results to record (replaces `RESULTS_PENDING` in the new doc)

From the sweep corpus, tabulate per voltage stratum and resistance: legacy-NLM top-1/top-3 localization
rate, balanced WLS alarm rate at sigma 0.01 (chi-square 0.01 or normalized residual 4.0, the 2026-09-14
detector), and the healthy-control alarm rate. From the main corpora: band composition, `expected_detectability`
counts, and how many rows the scenario generator admits. State explicitly that quiet detection-limit rows are
evaluation cases and that the legacy corpora were not modified. Update the memory note
`ieee14-hif-physical-units-20260919` with the final numbers.

## 5. Pitfalls
- Unit trap: any physical kV reaching `inject_midspan_hif_ieee14`, `constant_impedance_hif_kw`, `hif_ohms_from_pu` or `run_ieee14_hif_nlm(r_hif_ohm=...)` produces a fault thousands of times too weak; those take MODEL ohms (`r_hif_model_ohm`). The injector guard should raise; keep it.
- `r_hif_ohm` means model ohms in legacy labels and in `_simulate_candidate` output, physical ohms in new labels and estimator payloads; always go through `hif_units.label_*` helpers.
- Estimator convention must match the observation: a legacy scan (no marker) simulated with `ybus` would carry a constant 19 sigma offset at Qinj bus 9 and fail acceptance.
- Changing `CANONICAL_POWER_TOOLS` changes registry hashes; remove pins, do not regenerate gate rows.
- Do not repoint `CURRENT_TELEMETRY_HIF_SAMPLE_PATHS`; resumed legacy cells compare recorded sources.
- Run heavy OpenDSS tests one file at a time with long timeouts; the earlier session died on rate limits while six agents ran in parallel.
