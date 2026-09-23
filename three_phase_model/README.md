# Registry-driven three-phase OpenDSS models

Build a normalized balanced snapshot and physical load-unbalance example:

```powershell
python scripts/build_three_phase_model.py --system case57 --output-dir generated/ieee57_new
```

Use `--assumptions coupled_sensitivity` for the separately documented line sequence completion. The exporter requires an explicit supported assumptions schema and preserves canonical branch asset IDs, including parallel and nominal-tap transformers.

For the named IEEE57 physical reconstruction (buses 1–17 at 138 kV and 18–57 at 69 kV, on 100 MVA), use:

```powershell
python scripts/build_three_phase_model.py --system case57 --voltage-profile ieee57_reconstruction_138_69kv_v1 --output-dir generated/ieee57_physical_new
```

The voltage map is an explicit reconstruction because the canonical case does not supply physical bus ratings. Omitting `--voltage-profile` retains the normalized model. See [physical-ohm HIF configuration and WLS evidence](../docs/ieee57_physical_hif_20260919.md) for local resistance conversion, paired noise/covariance settings and detection limits.

`export_model()` returns the independently solved PYPOWER reference, asset registry and assumptions. `compile_model()` uses a new DSS context. `extract_measurements()` exports phase, sequence, total-power and canonical WLS measurements. `validate_model()` reads actual compiled YPrim, voltages and currents. `redistribute_load()` changes phase demand while preserving total demand and solves the resulting circuit.

The generated model uses fixed-PQ generator snapshots, explicit source sequence impedances and grounded-wye transformers. Zero-sequence values are research assumptions. The package does not claim AVR, nonlinear arcing HIF or harmonic equivalence, and does not enable the existing IEEE14-only diagnostic estimators for IEEE57.

See [model specifications and executed validation](../docs/ieee57_opendss_model_20260910.md) for results, limitations, MATLAB 8.1 reference execution, and the explicitly separated canonical versus legacy bus-shunt measurement conventions.

## Resistive HIF and unbalance experiments

```powershell
python scripts/validate_ieee57_disturbances.py --output-dir output/ieee57_disturbances_new --preset full --workers 4
```

The experiment builds four fresh IEEE57 models: diagonal/coupled assumptions at 0.8/1.0 load. It covers all 63 eligible lines and all 42 load buses. Every HIF has a paired no-fault split and removal/restoration checks. No faulted or weakly observable case is filtered from the results. Use `--preset smoke` for a small coverage check.

`disturbances.inject_midspan_hif()` splits only the series impedance, retains the original full-matrix charging at the external endpoints, and inserts an actual OpenDSS phase-to-ground resistor. `set_hif_enabled()` toggles the resistor; `restore_midspan_hif()` restores the original line. Both fault and paired healthy measurements retain the original 57-bus, 80-branch, 491-channel registry using `extract_measurements(..., branch_overrides=receipt['branch_overrides'])`. `audit_disturbed_circuit()` separately checks the hidden fault node and full circuit equations.

`diagnostics.capture_nominal_model()` requires a pristine model. `screen_measurements()` consumes only external phase voltage/current phasors and that nominal model. It ranks all branch and nodal residuals, propagates assumed sensor uncertainty through full ABC matrices, and estimates HIF distance/resistance when sufficiently observable. This is a new constitutive/nodal diagnostic; the IEEE14 NLM and policy routing are not silently reused. The standalone experiment separately runs the existing balanced WLS with both chi-square and normalized-residual gates.

Results include exact/noisy telemetry, immutable physical-root fingerprints, standalone DSS replay files, source hashes, all failures and non-detections, and a readable report. See the [executed experiment](../output/ieee57_hif_unbalance_20260911_verified/report.md) and [methods, solver fix, and scope](../docs/ieee57_disturbance_testing_20260911.md). These are fundamental-frequency resistive surrogates, not nonlinear arcing or harmonic simulations. Full phase telemetry and the declared sequence/grounding model are assumed.
