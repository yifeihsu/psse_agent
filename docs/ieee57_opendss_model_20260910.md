# IEEE57 OpenDSS three-phase model

Built and executed two automated, positive-sequence-preserving realizations from the canonical repository `mcp_server/case57.m`:

- [Diagonal model](../generated/ieee57/Master.dss): normalized 1 kV line-to-line, 100 MVA three-phase base; diagonal line R/X/C matrices, matching the existing IEEE14 line assumption.
- [Coupled sensitivity model](../generated/ieee57_coupled/Master.dss): the same positive-sequence network with illustrative line completions R0=3R1, X0=3X1 and C0=0.5C1. These ratios are explicitly assumed, not recovered equipment data.

Each has **57 external buses, 171 phase nodes, 63 lines, and 17 transformers**. The parallel 4–18 transformers and both nominal-tap 24–25 transformers remain distinct. There are 126 individually addressable phase loads at 42 load buses, 18 phase generator devices representing six non-slack generators, three bus shunts, and one source representing the slack generator.

Each output directory contains `Master.dss`, component DSS files, `asset_registry.json`, `assumptions.json`, `measurement_layout.json`, `measurements.json`, `positive_sequence_reference.json`, validation reports, and a hash manifest. Branch asset IDs match the shared system registry; padded OpenDSS names such as `Transformer.br_0035` retain immutable source-row identity instead of relying on endpoint pairs.

## Modeling contract

The original IEEE57 voltage bases are unspecified. The exporter assigns a documented normalized realization to all buses, without changing source per-unit R/X/B or taps. On these bases, Zbase=0.01 ohm and phase power base=100/3 MVA. Each single-phase load/generator receives one-third of total kW/kvar and uses 1/sqrt(3) kV line-to-neutral; branch impedance is not divided by three.

Line matrices are formed from independently specified zero-sequence values with negative sequence equal to positive sequence. Total branch capacitance is passed to the OpenDSS pi element without pre-halving. Transformer taps are on the from winding; both windings are explicitly grounded wye, on the same total three-phase kVA base, with zero magnetizing, no-load-loss and anti-float terms. Transformer charging, when present in another supported input, is retained as tap-adjusted endpoint admittances. These choices follow the documented [Line matrix and capacitance interface](https://dss-extensions.org/dss-format/Line.html) and [Transformer winding/base interface](https://dss-extensions.org/dss-format/Transformer.html).

Loads and non-slack generators are constant PQ snapshots using solved positive-sequence generator outputs. Their declared constant-PQ envelope is 0.5–1.5 pu; validation checks actual powers so fallback to a different voltage model cannot silently pass. Capacitive bus shunts retain their voltage-squared behavior and sign. The [OpenDSS Load](https://dss-extensions.org/dss-format/Load.html) and [Generator](https://dss-extensions.org/dss-format/Generator.html) definitions distinguish these models from voltage regulation.

The source has explicit Z0=Z1=Z2=(1+j)×10^-6 pu. Its internal EMF is calculated analytically as the target slack voltage plus Z1 times the reference slack current. This compensates the finite source drop at the balanced snapshot without pretending the source is ideal. The same EMF and sequence impedances remain fixed during unbalance. [OpenDSS source documentation](https://dss-extensions.org/dss-format/Vsource.html).

This establishes snapshot equivalence. PV/AVR response, reactive-limit enforcement, real equipment ratings, explicit neutral/earth geometry, arcing HIFs, and a validated frequency-dependent harmonic model are not established. Assigned spectra contain only the fundamental; no harmonic emissions are silently inherited from generator/load defaults. The model does not enable IEEE57 HIF/NLM diagnostic routing or claim those existing IEEE14 locators have been generalized.

## Executed balanced validation

The validator reads actual compiled OpenDSS terminal admittances and currents. It independently constructs the expected MATPOWER branch admittances from R/X/B/tap data, rather than validating exporter metadata against itself.

| Check | Diagonal model maximum error | Coupled model maximum error | Acceptance limit |
|---|---:|---:|---:|
| Bus voltage magnitude | 2.27e-13 pu | 8.24e-11 pu | 1e-6 pu |
| Wrapped voltage angle | 1.94e-8 deg | 1.94e-8 deg | 1e-4 deg |
| Branch P/Q, both ends | 2.73e-12 pu | 2.73e-12 pu | 1e-5 pu |
| Positive-sequence branch terminal admittance | 1.07e-14 pu | 1.13e-14 pu | 1e-8 pu |
| Phase-node KCL | 1.60e-10 pu | 1.43e-10 pu | 1e-7 pu |
| Canonical 491-channel measurement vector | 6.54e-11 pu | 1.94e-11 pu | 1e-5 pu |

All **23 balanced checks pass** for both configurations, including source/generator/load/shunt powers, losses, power balance, source sequence impedances, phase sequence nulls, component current equations, and complete asset/terminal coverage. Reports: [diagonal](../generated/ieee57/validation_report.json), [coupled](../generated/ieee57_coupled/validation_report.json).

An independent reference was also executed in **MATLAB R2026a / MATPOWER 8.1**, using [`mp.case_utils.convert_1p_to_3p`](https://matpower.org/documentation/ref-manual/classes/mp/case_utils.html). Its positive-sequence versus three-phase comparison passed all 14 checks at load scales 1.0 and 0.8. The [MATLAB script](../scripts/build_ieee57_matpower_3p_reference.m) preserves branch identities and checks typed line/transformer tables, generator/shunt powers, sequence nulls and KCL. The independently generated MATLAB three-phase case and results are copied into each model's `matpower_reference` directory.

Direct OpenDSS-versus-MATPOWER-three-phase checks also pass: [diagonal cross-reference](../generated/ieee57/matpower_reference/opendss_cross_reference_report.json) and [coupled cross-reference](../generated/ieee57_coupled/matpower_reference/opendss_cross_reference_report.json). A separately rebuilt [0.8-load OpenDSS model](../output/ieee57_opendss_scale080_20260910/validation_report.json) passes against the corresponding independent MATPOWER result. These are fresh solver executions, not replayed prior IEEE14 telemetry or OPF feasibility certificates.

## Actual unbalance and restoration

`UnbalanceExample.dss` reloads the healthy model and redistributes bus 12 phase demand by [1.2, 0.8, 1.0]. Total demand remains **377 MW + 24 Mvar** at unit load. OpenDSS solves the modified circuit; voltage/current phasors are extracted from that solution.

- Diagonal model: maximum |V2|/|V1| = **3.5686%**.
- Coupled sensitivity model: maximum |V2|/|V1| = **6.0967%**.

Both pass unbalanced phase KCL, component constitutive equations, device-power and network-power checks. Restoring the phase loads recovers the balanced voltage solution within 2.14e-12 pu. Standalone compilation of both `UnbalanceExample.dss` files reproduces the reported VUF values. Full phasors and checks are in `unbalance_measurements.json`, `unbalance_validation_report.json`, and `restoration_report.json` beside each master file.

The differing VUF demonstrates sensitivity to the explicitly assumed missing sequences; it does not identify the original IEEE57 zero-sequence network. Fault segmentation/recovery and harmonic corpus generation remain later extensions requiring their own paired references and physics validation.

## Measurement conventions

The canonical 491-channel layout is `[Vm(57), Pinj(57), Qinj(57), Pf(80), Qf(80), Pt(80), Qt(80)]`. Vm uses phase-A line-to-neutral magnitude, matching the legacy IEEE14 exporter. P/Q use total three-phase powers on 100 MVA. Branch terminal currents point into the branch, with phase bases derived from the declared system base; phase/sequence voltages, currents and powers are also exported separately.

Inspection found an existing IEEE14 convention mismatch: its actual exporter includes capacitor injection in bus Pinj/Qinj, while the balanced WLS measurement function uses generation minus demand excluding bus shunts. The new `measurement_vector` matches WLS; `legacy_ieee14_compatible_vector` preserves the actual IEEE14 convention. Both are explicitly named in the layout. They differ at the three IEEE57 shunt buses (18, 25, 53), even under balance. This difference must be chosen deliberately for downstream transfer experiments.

Total ABC power and 3V1I1* are separate exports under unbalance. Their distinction, ground-node exclusion, sign/base conversion, parallel branches, and registry-driven ordering are covered by tests.

## Reproduce and extend

Compile `generated/ieee57/Master.dss` in OpenDSS for the balanced model, or `UnbalanceExample.dss` for its phase-redistribution example. To rebuild into a new directory from the repository root:

```powershell
python scripts/build_three_phase_model.py --system case57 --output-dir generated/ieee57_new --assumptions normalized_diagonal
python scripts/build_three_phase_model.py --system case57 --output-dir generated/ieee57_coupled_new --assumptions coupled_sensitivity
```

Use `--matpower-reference-dir output/ieee57_matpower3p_reference_final_20260910` to additionally compare against and package the executed unit-load MATLAB reference. `--load-scale`, `--unbalance-bus`, and `--unbalance-delta` parameterize fresh builds. Existing output directories are preserved; unsupported assumptions, nonzero phase shifts, missing active slack/PV generators, invalid statuses and disconnected inputs are rejected explicitly.

The reusable implementation lives in `three_phase_model/`; the two JSON specifications are under `three_phase_model/specs/`. Validation passed **43 focused tests** covering exporter, actual compiled-engine checks, measurement conventions and build-stage gates. The generic exporter was also checked on IEEE14 and a small tapped/charging/shunt case without editing the existing IEEE14 testbed. Engine versions and generated-file/source hashes are recorded in each build manifest.
