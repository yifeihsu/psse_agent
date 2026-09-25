# IEEE118 three-phase OpenDSS model and fault-scenario migration

The IEEE118 testbed follows the IEEE57 route (docs/ieee57_opendss_model_20260910.md,
docs/ieee57_physical_hif_20260919.md): a canonical balanced case in the system
registry, registry-driven three-phase OpenDSS realizations, and the same fault
scenarios. Two findings force departures from the IEEE57 contract, and both are
described below: OpenDSS cannot hold the IEEE118 operating point on its own,
and fixed-PQ generator snapshots have no steady state for most strong HIFs.

## Canonical case

`mcp_server/case118.m` is PYPOWER 5.1.19 `pypower/case118.py` (SHA256
`90c28f0d…c25`) converted without numeric changes; both repository parsers
reproduce the source matrices exactly. The registry (`psse_env/systems`) resolves
`case118`/`ieee118`/`118`: 118 buses, 186 branches (177 zero-tap, 9 tapped
transformers, 7 parallel pairs), 54 generators with slack at bus 69, 14 bus
shunts (reactors at 5 and 37). The WLS layout has 1,098 channels, 235 states and
863 residual degrees of freedom. As for IEEE57, the registry supports the five
balanced families (no_error, measurement, multi_measurement, parameter,
measurement+parameter). No hash pin was added.

## Three-phase realizations

`scripts/build_three_phase_model.py --system case118` exports and validates:

| Directory | Voltages | Generators | Independent reference |
|---|---|---|---|
| `generated/ieee118` | normalized 1 kV | fixed PQ per phase | MATPOWER 8.1 3-phase conversion, load 1.0 |
| `generated/ieee118_coupled` | normalized 1 kV, R0=3R1, X0=3X1, C0=0.5C1 | fixed PQ | same |
| `output/ieee118_opendss_scale080_20260924` | normalized, load 0.8 | fixed PQ | MATPOWER 8.1, load 0.8 |
| `generated/ieee118_physical` | source BASE_KV | fixed PQ | PYPOWER reference |
| `generated/ieee118_physical_pv` | source BASE_KV | PV with reactive limits | limited PYPOWER reference = MATPOWER `enforce_q_lims` |
| `generated/ieee118_pv` | normalized 1 kV | PV with reactive limits | same |

Each model has 354 phase nodes, 186 branch assets, 297 phase loads at 99 load
buses and 14 shunts. The physical profile `ieee118_source_basekv_138_161_345kv_v1`
is the case's own BASE_KV column (345 kV at buses 8, 9, 10, 26, 30, 38, 63, 64,
65, 68, 81; 161 kV at bus 87; 138 kV elsewhere), unlike IEEE57, whose bases had
to be reconstructed. Two zero-tap branches join different source bases, 86-87
(138/161 kV) and 68-116 (345/138 kV). The exporter realizes them as ideal-ratio
transformers with the source charging at their endpoints, so the physical model
has 175 lines and 11 transformers. Those two branches are excluded from HIF
injection, leaving 175 eligible lines (165 at 138 kV, 10 at 345 kV).

All 24 balanced checks pass for every model (maximum errors):

| Check | diagonal | coupled | physical | physical PV | diagonal PV |
|---|---:|---:|---:|---:|---:|
| Bus voltage magnitude (pu) | 1.8e-13 | 1.4e-10 | 1.8e-13 | 2.9e-14 | 4.5e-14 |
| Voltage angle (deg) | 2.1e-8 | 2.5e-8 | 2.1e-8 | 2.1e-8 | 2.1e-8 |
| Branch P/Q, both ends (pu) | 1.2e-12 | 1.2e-12 | 1.1e-12 | 5.9e-13 | 5.7e-13 |
| Positive-sequence terminal Y (pu) | 1.2e-13 | 7.5e-14 | 2.3e-13 | 2.3e-13 | 1.2e-13 |
| Phase-node KCL (pu) | 7.7e-11 | 9.9e-11 | 9.2e-11 | 1.1e-10 | 1.8e-10 |
| 1,098-channel WLS vector (pu) | 4.0e-11 | 9.3e-11 | 1.0e-11 | 3.8e-11 | 7.0e-11 |

The MATPOWER 8.1 three-phase conversion (`scripts/build_matpower_3p_reference.m`,
a case-agnostic form of the IEEE57 script that reproduces the IEEE57 metrics
bit-for-bit) passed all checks at load 1.0 and 0.8. OpenDSS agrees with it to
1.8e-13 pu in voltage and 1.2e-12 pu in branch power. The bus-12 phase-load
example gives a VUF of 1.06% (diagonal) and 1.94% (coupled); restoring the loads
recovers the balanced solution to 6e-14 pu.

### OpenDSS cannot hold the IEEE118 operating point

DSS C-API 0.14.5 solves snapshots with a fixed-point current-injection iteration
(`Newton` and `Normal` give identical results; `NCIM` is not available). On
IEEE118 the operating point is a **repelling** fixed point of that iteration:
seeded exactly at the reference solution, the error grows roughly 2x per
iteration (1e-9 after 5 iterations, 1e-4 after 20). Compiled as-is, OpenDSS
reports convergence at a spurious state with bus 10 at 2.28 pu and loads on
their constant-impedance fallback. The weak link is the 450 MW generator at the
end of the heavily charged 345 kV corridor 8-9-10 (b = 1.16 and 1.23 pu). All
IEEE14/IEEE57 models solve normally.

`three_phase_model/runtime.py` therefore checks every device against its control
law after each OpenDSS solve (constant-PQ setpoints; P, Q limits and voltage
regulation for regulated generators). Only when a device fails does it solve the
compiled circuit by Newton-Raphson and write the state back. The Newton system
uses every linear element's own YPrim and the source's own Norton current, with
active-set reactive limits, globalized by a Newton homotopy from the pre-edit
state. OpenDSS's own iteration must then accept the state, and the existing
validators check it independently. Regulated-generator Q is seeded from KCL at the seed voltages, so a
restore after a failed fault solve cannot start from mismatched states. A
solution with any constant-power device outside its [Vminpu, Vmaxpu] band is
refused, because OpenDSS changes the device model there. On IEEE118 the
balanced compile takes two Newton steps and 0.1 s; a fault solve takes ~0.3 s,
and a refused one ~3.5 s. `solve()` uses the fallback only on a context whose
compile needed it. Every circuit OpenDSS solves natively (all IEEE14/IEEE57
models) keeps the original solve path. An earlier version read node arrays
before every solve, and in the full test suite that disturbed later
global-engine IEEE14 HIF estimator solves in the same process.

**WLS cost at IEEE118 size.** The balanced WLS port (`tools/lagrangian_port.py`)
formed the branch-multiplier covariance as `S (phi R phi') S'`, with a dense
1,098 x 1,098 diagonal R. It is now grouped as `(S phi) R (S phi)'`: the
returned arrays (multipliers, `ea`, normalized residuals, states, residual
covariance) agree with the previous implementation to <= 5e-14 relative on
IEEE14/57/118, and one IEEE118 fit drops from 9.3 s to 1.5 s (IEEE57 0.51 s to
0.20 s). This affects every IEEE118 WLS call, including the DAgger environment.

**Consequence:** an IEEE118 `Master.dss` opened in plain OpenDSS lands on the
spurious state. Use `three_phase_model.runtime.compile_model`/`solve`, which is
what every exporter, harness and sweep consumer already does.

### Generator control: fixed PQ has no steady state for strong HIFs

The IEEE57 contract freezes each non-slack generator at its snapshot P/Q on each
phase. On IEEE118 that operating point stops existing for moderate
single-phase faults. In the diagonal completion the phases decouple, so
existence was checked with PYPOWER on the faulted-phase network at every eligible
line and sweep resistance (unit load, midspan):

| R (ohm) | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed PQ: lines with an operating point | 10 | 88 | 121 | 157 | 165 | 175 | 175 |
| PV regulation, no limits | 170 | 173 | 175 | 175 | 175 | 175 | 175 |

(Load 0.8: fixed PQ 27/110/143/162/173/175/175.) With fixed Q, the charged 345 kV
corridor pushes bus 10 without bound (1.24 pu at 127 ohm on line 1-2, no
solution at 100 ohm).

**Decision (2026-09-25): IEEE118 disturbance scenarios use voltage-regulated
generators with reactive limits** (`export_model(..., generator_control="pv_q_limits")`,
`--generator-control pv_q_limits`). Each unit is one three-phase OpenDSS
`Model=3` generator: fixed P, equal Q on the three phases, Q adjusted to hold the
average phase-to-neutral magnitude at VG and clamped to [QMIN, QMAX]. The
reference solve enforces the same limits by the usual outer loop. It clamps
violators and releases a unit whose voltage is on the wrong side of its
setpoint, and the slack is an unlimited Thevenin source. At unit load units 19,
32, 34, 92, 105 sit at QMIN and 103 at QMAX; at load 0.8, 16 units are limited.
The limited reference equals MATPOWER 8.1 `pf.enforce_q_lims` to 3e-14 pu (V) and
6e-11 MVAr (Q). The IEEE118 balanced snapshot therefore differs deliberately
from the unlimited case118 power flow.

OpenDSS's own `Model=3` law agrees with this realization. From an accepted state
under a 100-ohm fault and under a 20% bus-12 load redistribution, further
OpenDSS iterations change Q by less than 1e-3 var and voltage by 5e-13.

The fixed-PQ models remain built for IEEE57-parity balanced checks.

**Remaining existence limit.** OpenDSS generator models deliver equal power on
every phase, so a regulated unit has no low negative-sequence impedance to hold
its phases together under a strong single-phase fault. Continuation in the fault
resistance itself (from 100 kohm downward in 10% steps, each solve seeded by the
last) gives the lowest resistance with an operating point, midspan, phase A,
unit load:

| Line | kV | Lowest R (ohm) | Below it |
|---|---:|---:|---|
| 1-2 | 138 | 65 | bus-10 unit's phase voltage leaves [0.5, 1.5] pu |
| 15-17 | 138 | 54 | same |
| 62-67 | 138 | 40 | no solution found along the homotopy |
| 80-97 | 138 | 95 | same |
| 44-45 | 138 | < 20 | still solvable at the 20-ohm floor |
| 9-10 | 345 | 437 | no solution found along the homotopy |

The Newton fallback refuses a solution that puts any constant-power device outside
its [Vminpu, Vmaxpu] band, where OpenDSS changes the device model. For line 1-2
at 50 ohm, Newton does converge, but to a state with the faulted phase at
2.04 pu at bus 10 and the healthy phases at 0.43 pu, and that state is rejected.
Such faults are recorded as physical failures with the reason, never filtered.
The 100-500 ohm band used by the IEEE14 HIF corpora solves on every sampled
138 kV line.

The phase screen (`three_phase_model/diagnostics.py`) is adapted to regulated
units. At a bus with a regulated generator, the common-mode reactive change is
fitted (GLS) and removed from the nodal current residual, with propagated
covariance, and only the active total enters the total-preserving unbalance
test. Healthy telemetry stays quiet, and a 100-ohm HIF on line 1-2 is localized
to the correct line and phase.

## Fault-scenario migration

| Family (IEEE57 route) | IEEE118 status | Evidence |
|---|---|---|
| Balanced: no_error, measurement, multi_measurement, parameter, measurement+parameter | enabled by `--system case118` on the corpus builder, round-0 aggregate, research runner and suite builder | fresh corpus 30/30 physically admitted; round-0 expert smoke identical per family to IEEE57 on the same plan and seed |
| Logical topology (branch-status and bus-coupler errors) | `scripts/validate_logical_topology.py --system case118` | smoke: 60/66 rows admitted (6 islanding rejections), 35 exact recoveries, 0 false corrections, 0 healthy-CB failures |
| Resistive HIF and phase-load unbalance, normalized harness | `scripts/validate_ieee57_disturbances.py --system case118` (PV, chi-square 1% or NR 4) | results below |
| Physical-ohm HIF detectability sweep | `scripts/audit_ieee118_hif_physical_sweep.py` (PV, 1% primary, 5% comparison) | results below |
| Harmonic, node/breaker topology, IEEE14 waveform corpora (PMU HIF estimators, multi-scan) | not migrated, as for IEEE57 | IEEE14-specific machinery |

**Balanced families.** `scripts/build_balanced_corpus.py --system case118` built
a 6/12/12 corpus (no_error / meter error / parameter error) with every AC-OPF
window physically admitted, in 16 s. The round-0 expert aggregate (canonical
protocol, `wls_gated_diagnostics`, chi-square 0.01 OR NR 4, 40 steps) was run
with the same plan and seed on IEEE57 and IEEE118. Every family ends the same
way on both: the clean root resolves, and each fault root is corrected (1, 1, 3
and 2 accepted corrections for measurement, parameter, multi-measurement and
mixed) and handed to the operator. The strict truth audit leaves the same five
checks open on every fault root of both systems. IEEE118 drew seven roots, one
fewer mixed root than IEEE57 (small corpus). Local runs need
`PSSE_LOCAL_DIAGNOSTIC_BUILD=1`. For an HPC cell, deploy with
`research/hpc/full_pipeline_20260907/overrides/ieee118_balanced_20260925.env`
(`SYSTEM=case118`, stage-0 corpus, the five balanced families at the IEEE57
sizes). Set `FROZEN_STUDENT_ADAPTER` there for a zero-shot transfer like the
IEEE57 run, or leave it empty to train BC0.

**Detector.** IEEE118 uses the pipeline detector chosen 2026-09-14 (chi-square
alpha 0.01 OR maximum normalized residual 4). The threshold is 962.58 at 863
degrees of freedom. With 1,098 channels, the NR-only gate carries more
healthy false alarms than on IEEE57; measure it from healthy controls rather
than assume it.

### Normalized HIF and unbalance harness

`output/ieee118_hif_unbalance_20260925` (seed 20260925, 2026-09-25): four freshly
built PV models (diagonal/coupled at load 0.8/1.0), every line and phase once
per model (177 x 3), every load bus at delta 0.05 and 0.20 (99 x 2), and 100
healthy noise replicates per model. R = 10/100/1000 pu and alpha 0.2/0.5/0.8
are cycled across assets, as in the IEEE57 study. All 3,316 rows ran without
execution failure. All 2,124 HIF splits, fault solves, removals and
restorations, and all 2,916 restorations, passed their physical checks. The
implementation was unchanged during the run.

| Family | Cases | Noisy WLS alarms | Phase correct, exact | Phase correct, nominal noise | Phase correct, precision |
|---|---:|---:|---:|---:|---:|
| healthy | 400 | 34 (2 chi-square, 32 NR-only) | 400 | 400 | 400 |
| HIF | 2,124 | 358 | 1,416 | 1,319 | 2,016 |
| unbalance | 792 | 311 | 718 | 728 | 792 |

| Subset | Cases | Noisy WLS alarms | Correct, nominal | Correct, precision |
|---|---:|---:|---:|---:|
| HIF R = 10 pu | 708 | 246 | 708 | 708 |
| HIF R = 100 pu | 708 | 58 | 611 | 708 |
| HIF R = 1000 pu | 708 | 54 | 0 | 600 |
| unbalance delta 0.05 | 396 | 75 | 336 | 396 |
| unbalance delta 0.20 | 396 | 236 | 392 | 396 |

No localization named a wrong line, phase or bus; 6 HIFs were ambiguous and the
rest of the misses were undetected. The healthy WLS false-alarm rate is 8.5%,
almost entirely from the normalized-residual gate over 1,098 channels, so the
R = 100 and 1000 pu WLS alarm rates (8.2%, 7.6%) are at the healthy level.
1,023 HIFs and 422 unbalance cases localize correctly from nominal-noise phase
telemetry while WLS stays quiet. This repeats the IEEE57 finding that a
WLS-only acquisition trigger misses most phase-localizable disturbances
(IEEE57: 252/252, 224/252 and 0/252 HIFs localized at 10/100/1000 pu). Given a
correct line and phase, the HIF distance fit has median |alpha error| 0.025
(95th percentile 0.33), and the resistance fit has median relative error 1.9%
(95th percentile 17.7%). These are conditional on correct localization, not
calibrated confidence statements.

### Physical-ohm HIF sweep

`output/ieee118_physical_hif_20260925/sweep` (seed 20260924, 20 shards,
57 min): all 175 eligible lines x 3 phases x 7 resistances x 2 load parents
(0.8, 1.0), midspan, one standardized noise group per line/phase/parent (1,050
groups) paired across resistance, noise profile and the healthy and no-fault
split controls. All 7,350 cases are recorded and the merged coverage validation
passed, with no failed controls, no WLS failures and no implementation change
during the run.

Operating points (solved cases / cases):

| kV | 50 | 100 | 200 | 500 | 1000 | 2000 | 5000 ohm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 138 | 558/990 | 831/990 | 909/990 | 987/990 | 990/990 | 990/990 | 990/990 |
| 345 | 0/60 | 0/60 | 27/60 | 60/60 | 60/60 | 60/60 | 60/60 |

The 828 missing operating points are retained as physical failures: 744 with no
solution along the homotopy and 84 whose solution leaves a device's
constant-power band.

WLS detection at baseline noise (sigma Vm 0.001, sigma PQ 0.01 pu) with the
pipeline gate (chi-square 1% OR NR 4). "New" counts HIF alarms whose paired
healthy observation with the same noise is quiet:

| kV | R (ohm) | Alarms / solved | New alarms | Shared with healthy |
|---|---:|---:|---:|---:|
| 138 | 50-500 | 100% | 93-95% | 5-7% |
| 138 | 1000 | 636/990 (64.2%) | 567 (57.3%) | 69 |
| 138 | 2000 | 169/990 (17.1%) | 101 (10.2%) | 68 |
| 138 | 5000 | 84/990 (8.5%) | 16 (1.6%) | 68 |
| 345 | 200-2000 | 100% | 93-96% | 4 |
| 345 | 5000 | 55/60 (91.7%) | 51 (85.0%) | 4 |

Healthy and no-fault-split controls alarm in 7.1% of 138-kV-paired and 6.7% of
345-kV-paired observations (10.1% and 6.7% under the 5% chi-square
comparison). With the 5% gate, 138 kV detection is 69.6% at 1000 ohm, 22.7% at
2000 and 12.3% at 5000 (IEEE57 at 138 kV: 59.6%, 13.5%, 6.4% against an 8.5%
healthy rate). The CSVs `detection_by_voltage_resistance*.csv` and
`controls_summary*.csv` give all three noise profiles.

On IEEE118, balanced WLS therefore sees physically solvable HIFs reliably
through 500 ohm at 138 kV and through 5000 ohm on the ten 345 kV corridor
lines. Around 2000 ohm at 138 kV the gain over healthy noise is small, and at
5000 ohm it is absent. These are synthetic coverage results under the diagonal
completion, PV-with-limits generators and two operating parents, not field
detection probabilities.

## Reproduce

```powershell
python scripts/build_three_phase_model.py --system case118 --output-dir generated/ieee118_new --matpower-reference-dir output/case118_matpower3p_reference_20260924
python scripts/build_three_phase_model.py --system case118 --voltage-profile ieee118_source_basekv_138_161_345kv_v1 --generator-control pv_q_limits --output-dir generated/ieee118_physical_pv_new
matlab -batch "addpath('scripts'); build_matpower_3p_reference('case118', 'output/case118_matpower3p_reference_new', 1.0)"
python scripts/build_balanced_corpus.py --system case118 --output-dir output/ieee118_balanced_corpus_new
python scripts/validate_logical_topology.py --system case118 --output-dir output/ieee118_logical_topology_new --preset smoke
python scripts/validate_ieee57_disturbances.py --system case118 --output-dir output/ieee118_hif_unbalance_new --preset full --workers 4
python scripts/audit_ieee118_hif_physical_sweep.py --output-dir output/ieee118_physical_hif_new --workers 20
```

Tests: `tests/test_ieee118_three_phase_model.py` (profile, solver fallback,
limited reference, PV control law, fixed-PQ non-existence, regulated phase
screen) plus the IEEE118 cases in `psse_env/systems/test_systems.py` and
`psse_env/examples/test_generate_round0_aggregate.py`.

## Scope

These are fundamental-frequency steady-state realizations. Sequence and
grounding data are research assumptions. Generator models are OpenDSS PQ or
average-magnitude PV with equal per-phase power, not machine sequence
impedances. HIFs are resistive surrogates without arcing or harmonics. No
learned policy was trained or evaluated on IEEE118.
