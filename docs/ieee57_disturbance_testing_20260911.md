# IEEE57 resistive HIF and phase-load unbalance tests

The test harness is [`scripts/validate_ieee57_disturbances.py`](../scripts/validate_ieee57_disturbances.py). Its full preset covers 756 HIF placements and 336 unbalance disturbances in four freshly compiled OpenDSS models: diagonal and coupled sequence completions, each at 0.8 and 1.0 load. It also evaluates 400 independent healthy instrument-noise realizations on four healthy physical roots. The experiment is engineering validation, not a frozen final test for model selection or a learned-policy evaluation.

- [Verified execution report](../output/ieee57_hif_unbalance_20260911_verified/report.md)
- [Machine-readable results](../output/ieee57_hif_unbalance_20260911_verified/summary.json)
- [Initial run, including a rejected split-null solve](../output/ieee57_hif_unbalance_20260911/summary.json)

## Verified results

All 1,092 disturbance solves passed the required circuit, paired-control and restoration checks after the initialization fix. There were zero execution failures. The independent audit checked all 1,492 acquisition files, each containing three measurement profiles, and reproduced 17 standalone DSS scenarios, including the previously rejected case. The implementation passed 175 focused tests (160 model/experiment tests and 15 artifact-auditor negative tests).

| Family | Physical disturbances | Correct localization, nominal phase noise | Correct localization, tenfold lower phase noise |
|---|---:|---:|---:|
| Resistive HIF: branch and phase | 756 | 476 | 728 |
| Load unbalance: bus | 336 | 209 | 325 |

No unambiguous branch/phase or bus localization selected the wrong target. Remaining cases were undetected or ambiguous. At nominal noise, all 252 R=10 pu HIFs localized, 224/252 R=100 pu localized, and 0/252 R=1000 pu localized. The precision profile localized 224/252 of the weakest HIFs. Of the 11 precision-profile unbalance misses, eight were at source bus 1 and three were weak delta=0.05 cases at bus 32.

Balanced WLS produced 42 alarms in 400 healthy-noise scans: 24 chi-square alarms plus 18 additional normalized-residual-only alarms. Phase diagnostics produced zero alarms on these healthy controls. Conversely, 334 HIFs and 154 unbalance cases localized correctly from nominal-noise phase observations while WLS had no alarm. A WLS-only acquisition trigger would therefore miss 488 phase-localizable disturbances in this corpus.

Correct HIF branch/phase identification is distinct from accepted distance/resistance estimation. At nominal noise, 283 of the 476 correct candidates passed the parameter uncertainty gates. Their conditional median absolute alpha error was 0.0101 and the 95th percentile was 0.0844; median relative resistance error was 1.11%, with a 95th percentile of 12.34%. These figures describe the accepted subset and are not an unconditional or empirically calibrated confidence guarantee.

- [Independent artifact, physics and replay audit](../output/ieee57_hif_unbalance_20260911_verified/artifact_audit.json)
- [Independent WLS and phase-statistics audit](../output/ieee57_hif_unbalance_20260911_verified/independent_wls_phase_audit.json)

The executed engine was OpenDSS C-API 0.14.5 through OpenDSSDirect.py 0.9.4 and DSS-Python 0.15.7. Current and preserved implementation hashes match the verified run.

## Physical scenarios and controls

HIF means the review's steady-state phase-to-ground resistive surrogate. All 63 eligible lines receive one fault on each phase per model. Resistance cycles through 10, 100 and 1000 pu, and position alpha through 0.2, 0.5 and 0.8. This is coverage of all assets and phases with varied severities, not the full Cartesian product of all positions, phases and resistances on every asset.

The normalized model has a 1 kV line-to-line, 100 MVA three-phase base, so impedance base is 0.01 ohm. The tested fault resistances are consequently 0.1, 1 and 10 ohm in the normalized circuit. They are not directly comparable with fault resistances in a transmission model at a different voltage base. Alpha measures a fraction of the equivalent branch series impedance; the case does not supply geographic line distances.

The injector divides the full ABC series impedance into alpha and 1-alpha sections. It keeps the original half-charging matrices at the two external endpoints, with no added hidden-node charging. A no-fault Kron reduction therefore recovers the original full ABC terminal admittance. The original 80 branch identities remain intact in measured terminal currents and the 491-channel operator vector.

Unbalance changes one load bus's phase demands by factors `[1+delta, 1-delta, 1]`, with delta 0.05 or 0.20, at all 42 load buses. Total active and reactive demand remain unchanged. OpenDSS solves each modified circuit; no exported phasor is manually changed to manufacture a physical disturbance.

All cases check full circuit KCL, passive current equations, device and network powers, and constant-PQ operation. HIF cases additionally check actual resistor current and power, paired no-fault splitting, fault removal and restoration of the original line. Hidden fault-node quantities are available only to the offline physics audit, not to the diagnostic input.

## Solver issue caught by the initial sweep

The first complete sweep retained one rejected case: `coupled_sensitivity_080_hif_0190`, canonical branch row 42 (asset `case57:branch:43`), alpha 0.2, phase B. Before the resistor was enabled, OpenDSS reported convergence at a different low-voltage solution. The passive split admittance still matched within about 1e-15 pu and KCL passed, but external voltage differed by 0.709 pu and device powers showed that loads had left their constant-PQ range. Thus convergence and KCL alone were insufficient evidence for accepting the scenario.

The injector now captures the actual preceding circuit solution, initializes all original nodes by name, and initializes the new no-fault series node by the exact interpolation `(1-alpha)*Vfrom + alpha*Vto`. It then runs the real OpenDSS solve and checks that the external healthy solution is recovered before enabling the resistor. These are numerical solver initial values; measurement values are extracted only after solving. Standalone DSS replay uses `CalcVoltageBases` and a no-fault solve before enabling the fault. Fault toggling and restoration also retain a consistent preceding numerical state.

The original run and its matching implementation snapshot are retained. The verified directory contains a complete rerun of the same predefined scenarios after the initialization fix; it must not be counted as another set of independent physical roots. Physical acceptance tolerances and diagnostic detection thresholds were unchanged.

## Diagnostic interpretation

The balanced WLS provider runs from a flat voltage initialization, with sigma(Vm)=0.001 pu and sigma(P/Q)=0.01 pu. For 491 measurements and 113 states, the chi-square threshold is 424.334166 at alpha=0.05 and 378 degrees of freedom. An alarm also occurs when the maximum normalized residual reaches 4. Every scan records both tests, including alarms caused only by normalized residuals.

The new phase diagnostic consumes external ABC bus voltages and both-end ABC branch currents. It uses all 80 nominal branch admittance matrices, plus nominal device powers and shunts, to rank branch and nodal equation residuals. It does not receive fault labels, hidden-bus measurements, changed load settings, or exported device-power labels. All measurements are available in this offline diagnostic experiment; a learned policy's acquisition decisions were not executed.

The nominal phase-instrument model uses independent Gaussian noise per real/imaginary component: voltage sigma 1e-4 pu and current sigma 1e-3 pu. A separate precision sensitivity profile reduces both by ten while reusing the standardized noise draws. SCADA noise is identical between those two phase profiles. The fixed phase residual gate is six times the propagated component noise scale, not an asserted global six-sigma false-alarm probability. Exact phase telemetry is still evaluated with the nominal sensor weights and gate.

Correct HIF localization means the correct branch and phase. Raw distance and resistance fits are reported separately with first-order uncertainty and an explicit parameter-acceptance flag. A correct branch candidate can still have an unreliable or out-of-range distance estimate. Parameter-error distributions conditional on a correct branch must not be read as accepted-parameter accuracy.

Weak faults may remain below instrument resolution. Source-bus load redistribution is especially weakly observable with the measured channels used here: estimating source current from a stiff Thevenin voltage relation amplifies voltage uncertainty. Measuring source phase currents would supply additional information; this experiment does not assume that channel exists.

This validates the declared fundamental-frequency model and its new constitutive/nodal diagnostic. Nonlinear arcing, harmonic emissions, transient behavior, sparse sensors, uncertain sequence/grounding data, the legacy IEEE14 NLM, and learned-agent performance on IEEE57 are outside the executed test scope.

## Reproduction

```powershell
python scripts/validate_ieee57_disturbances.py --output-dir output/ieee57_disturbances_new --preset full --workers 4 --seed 20260911
python scripts/audit_ieee57_disturbance_artifacts.py --output-dir output/ieee57_disturbances_new --extended-replays
python scripts/audit_ieee57_disturbance_statistics.py --output-dir output/ieee57_disturbances_new
```

Each model directory contains its DSS model, balanced reference checks, nominal diagnostic model, scenario definitions, all result rows, compressed truth-free exact/noisy telemetry, and standalone scenario DSS files. The root directory retains experiment settings and SHA-256 snapshots of the implementation. Failed, ambiguous and undetected cases are retained rather than filtered.

Relevant component/API documentation: [OpenDSS Fault](https://dss-extensions.org/dss-format/Fault.html), [Capacitor matrix units](https://dss-extensions.org/dss-format/Capacitor.html), and [OpenDSSDirect YMatrix API](https://dss-extensions.org/OpenDSSDirect.py/opendssdirect.html).

The historical study above used a normalized 1 kV realization without a declared physical bus-kV map. The later [138/69 kV reconstruction and fresh physical-ohm results](ieee57_physical_hif_20260919.md) are separate evidence.
