# IEEE14 HIF experiments with declared physical voltage bases

The default versioned scenario bundle now uses `ieee14_physical_hif_v1`.
It adopts the supplied nominal-voltage map as an explicit model realization:

| Buses | Line-to-line voltage base |
| --- | ---: |
| 1–5 | 69 kV |
| 6–7, 9–14 | 13.8 kV |
| 8 | 18 kV |

This map agrees with the IEEE14 bus table in the supplied
[Manchester dataset](https://pure.manchester.ac.uk/ws/portalfiles/portal/184633155/FULL_TEXT.PDF).
The checked-in MATPOWER source omits these bases, so the code applies a named,
versioned map to a copy. It does not edit the canonical case or assume every
published IEEE14 variant has identical voltage metadata.

## Units and fault populations

Every fault uses its own line's line-to-line base and the common 100 MVA
three-phase power base:

`Zbase_ohm = Vbase_kV_LL**2 / Sbase_MVA_3ph`

`Rfault_pu = Rfault_ohm / Zbase_ohm`.

| Physical resistance | 69 kV, Zbase=47.61 ohm | 13.8 kV, Zbase=1.9044 ohm |
| --- | ---: | ---: |
| 50 ohm | 1.0502 pu | 26.2550 pu |
| 100 ohm | 2.1004 pu | 52.5100 pu |
| 200 ohm | 4.2008 pu | 105.0200 pu |
| 500 ohm | 10.5020 pu | 262.5499 pu |
| 1000 ohm | 21.0040 pu | 525.0998 pu |
| 2000 ohm | 42.0080 pu | 1050.1995 pu |
| 5000 ohm | 105.0200 pu | 2625.4988 pu |

The 18 kV impedance base is 3.24 ohm. Bus 8 has no same-voltage line suitable
for the existing midspan-line HIF injection; its 7–8 connection is a
13.8/18 kV transformer even though MATPOWER encodes its nominal pu tap as zero.
The new model has 16 eligible lines (seven at 69 kV, nine at 13.8 kV) and four
transformers. Transformer faults are outside this line-fault population.

The main training proposal uses **100–1000 ohm on 69 kV lines**, split into
100–200, 200–500, and 500–1000 ohm bands. Early curriculum representation is
2:2:1; the full curriculum is 1:1:1. Sampling is uniform within each physical
resistance band, with all phases eligible and location uniform over 0.25–0.75.
This realizes 2.1004–21.0040 pu, close to the requested approximate 2–20 pu range.

Evaluation includes 50, 100, 200, 500, 1000, 2000, and 5000 ohm at both eligible
voltage levels. These cases are retained independently of WLS detection. The
new `hif_resistance_evaluation_manifest.jsonl` is an ohmic/voltage-stratified
sweep, not a relabeling of the previous 20–200 pu weak-HIF artifact. The same
ohms are 25 times larger in pu at 13.8 kV, so lower-voltage faults must not be
described as the 69 kV 2–20 pu population.

The physical interval is a research choice supported by examples of resistive
fault studies, not a universal HIF definition. The supplied
[FFT study](https://www.ijert.org/application-of-fft-for-detection-of-high-impedance-faults-in-power-system-network)
uses 500 ohm simulation faults. The utility-authored
[SDG&E/WPRC study](https://wprcarchives.org/wp-content/uploads/2024/04/Udren_Tariq_Transmission-Line-Falling-Conductor-Protection-System-Development-at-SDGE_20220909.pdf)
reports tests extending into several kilohms; its PMU protection performance
is not transferred to this balanced-WLS experiment.

## Physical implementation

The registry-driven exporter uses local voltage ratings for loads, generators,
bus shunts, source impedance, line R/X/C, and transformer winding voltages.
Native per-unit branch impedances, charging, and taps are preserved. The 7–8
transformer uses a nominal per-unit tap of 1.0 with 13.8/18 kV winding ratings.
Explicit bus bases follow `CalcVoltageBases` so the 18 kV bus is not silently
assigned a different preferred OpenDSS voltage.

The HIF injector accepts either `resistance_ohm` or `resistance_pu`; specifying
both is rejected. Physical-profile replay settings contain only ohms. Receipts
record both units, local voltage/current/impedance bases, and solved fault-point
current, voltage and dissipation. The engine's Fault element is a constant
single-phase resistor to ground; it does not implement nonlinear arcing.

Physical validation converts each terminal's admittance using
`Ypu_ij = Ysi_ij * Vbase_j / Ibase_i`. KCL, passive-device equations, transformer
currents and hidden-fault-node checks use their respective local bases. The
analytic phase diagnostic reports resistance and uncertainty in ohms using
the fault candidate's local base. No legacy 5 pu lower bound is imposed on
the new physical injection or analytic estimate.

Changing voltage bases consistently does not itself change a per-unit WLS
solution. The exporter is checked against the unchanged positive-sequence
model. Detection changes arise from the newly specified local-pu fault
resistance, operating point and observation noise—not from relabeling volts.
All noise/covariance profiles and the existing WLS thresholds remain unchanged.

## Generate and inspect

```powershell
python -m research.reviewed_fault_scenarios --output-dir output/new_ieee14_physical_bundle --scenario-profile ieee14_physical_hif_v1 --seed 20260918
```

The physical profile is now the bundle CLI default. Use `--scenario-profile
reviewed_v1` explicitly to reproduce the prior normalized configuration.
The lower-level exporter retains its old normalized default for compatibility;
pass `voltage_profile="ieee14_nominal_69_13p8_18kv_v1"` for the new realization.

The existing training admission remains active: actual noisy WLS evidence and
an executed observable expert prefix are required. Physical-only and meter-only
checks still apply to mixed scenarios. New profile identity and voltage bases
survive graph construction, auxiliary-sensor replay, canonical SFT export and
the training admission guard. Bus voltage bases are public network metadata;
no fault resistance or fault-location truth enters the learned graph.

Historical normalized corpora and their saved currents/ohms are unchanged.
The legacy IEEE14 NLM fitting path is not silently certified for the new
registry-driven physical models; current training admission still certifies
an expert prefix, not a complete HIF-location/repair episode. Grounded-wye
connections, sequence completion and constant-PQ controls remain declared
research assumptions rather than validated field models.

## Executed validation and detection sweep

Fresh results are under `output/ieee14_physical_hif_20260918/`. The affected
regression selection passed **431 tests and six subtests**. The four-parent
`physical_bundle` generated the new voltage/ohm profile and retained 32/38,
37/38 and 38/38 training observations under the baseline/0.005/0.002 power-noise
profiles. All 126 exported expert-prefix targets passed the canonical chat and
teacher-realizability checks. One unrelated requested unbalance slot in this
small bundle reached its attempt cap; that shortfall remains in its report.

The independent unfiltered sweep covered two load parents (0.8 and 1.0), all
16 eligible lines, all three phases, seven resistances, and fixed midspan
location 0.5: **672/672 physical cases passed**. It evaluated 2,016 noisy fault
observations and 576 healthy/no-fault control observations, with zero circuit
or WLS failures. The 96 standardized-noise groups are paired across resistance,
accuracy and controls; these observations are not independent operating parents.

Maximum all-node KCL error was 2.51e-10 pu. Independently injecting ohms and the
corresponding local pu resistance changed external measurements by at most
1.74e-14 pu. Neither physical tolerances nor detector thresholds were relaxed.

| 69 kV resistance | WLS alarms, power sigma 0.01 | sigma 0.005 | sigma 0.002 |
| --- | ---: | ---: | ---: |
| 50 ohm | 42/42 | 42/42 | 42/42 |
| 100 ohm | 42/42 | 42/42 | 42/42 |
| 200 ohm | 42/42 | 42/42 | 42/42 |
| 500 ohm | 1/42 | 32/42 | 42/42 |
| 1000 ohm | 0/42 | 1/42 | 41/42 |
| 2000 ohm | 0/42 | 0/42 | 6/42 |
| 5000 ohm | 0/42 | 0/42 | 1/42 |

Lower-voltage faults are substantially weaker under the same absolute power
noise. At 13.8 kV, the baseline alarms at 1000, 2000 and 5000 ohm are each 1/54,
but all are the same noise group already alarming when healthy: zero newly
alarming windows. The paired healthy and no-fault controls each have 1/96
baseline alarms and 0/96 under the two accuracy profiles. The complete
voltage-stratified results are in `sweep/detection_by_voltage_resistance.csv`.

Thus physical voltage-base correctness and physical fault plausibility do not
guarantee balanced-WLS detection. Quiet physical cases remain evaluation cases;
the training-only observability/actionability filter remains in force. No
neural model was retrained and no full-episode recovery claim is made.

## 2026-09-19 additions: resistance classification, detection-limit cohort, default rule

These additions change only the physical (`ieee14_physical_hif_v1`) stack.
The reviewed and legacy profiles, their pu samplers and every tracked corpus
are unchanged. Profile edits change `profile_identity_sha256` in freshly
generated manifests; the saved `output/ieee14_physical_hif_20260918/` bundle
retains the hashes it recorded on 2026-09-18.

### Physical resistance classification (ohms, voltage-agnostic)

Source of truth: `three_phase_model.voltage_bases.HIF_RESISTANCE_CLASSES_OHM`,
`hif_resistance_class(r_ohm)` and `hif_resistance_classification_table()`.
The profile exposes the same table as
`profile["hif"]["resistance_classification_ohm"]` (lazily imported, like the
voltage-base profile). Lower bounds are inclusive, upper bounds exclusive, and
the final class is open, so the shared endpoints at 100, 200, 500, 1000 and
5000 ohm each belong to exactly one class and 100–200 ohm is not a gap.

| Class | Bounds (ohm) | 69 kV pu (Zbase 47.61 ohm) | Interpretation |
| --- | --- | --- | --- |
| `low_resistance_fault` | [0, 50) | [0, 1.05) | low or moderate fault resistance; not treated as an HIF |
| `moderately_resistive` | [50, 100) | [1.05, 2.10) | moderately resistive fault (69 kV: ~400–800 A) |
| `moderately_high_resistance` | [100, 200) | [2.10, 4.20) | entering HIF territory (69 kV: ~200–400 A) |
| `representative_hif` | [200, 500) | [4.20, 10.50) | representative HIF (69 kV: ~80–200 A) |
| `weak_hif` | [500, 1000) | [10.50, 21.00) | weak HIF (69 kV: ~40–80 A) |
| `extreme_weak_hif` | [1000, 5000) | [21.00, 105.02) | extreme or very weak HIF; detection-limit population (69 kV: ~8–40 A) |
| `near_open_circuit` | [5000, inf) | [105.02, inf) | near-open-circuit downed conductor; a few amperes at 69 kV |

The classification applies to physical ohms only. It is not a per-unit
severity: 500 ohm is 10.5 pu at 69 kV but 262.5 pu at 13.8 kV and both are
`weak_hif`; report `hif_units` (local pu on the faulted line's base) alongside
the class. Every physical HIF manifest row now carries
`offline_metadata.settings.hif_resistance_class`, and every injector receipt
carries `resistance_class` with a `resistance_class_scope` note (on a
normalized 1 kV registry the classified ohms are model ohms, not physical).

Unit caution recorded on the profile as `pu_equivalents_69kv`: the pu figures
are derived labels on the 69 kV / 100 MVA base and are not the sampled
quantity. **1000 pu at 69 kV is 47.6 kOhm** (a near-open circuit) and must not
be described as a 1000 ohm HIF; 1000 ohm at 69 kV is 21.0 pu. The main range
100–1000 ohm is 2.10–21.00 pu, the sweep 50/100/200/500/1000/2000/5000 ohm is
1.05/2.10/4.20/10.50/21.00/42.01/105.02 pu, all on the 69 kV base only.

### Detection-limit cohort (evaluation only)

`profile["hif"]["detection_limit_band_ohm"] = [1000.0, 5000.0]` (21.0–105.0 pu
at 69 kV) with `detection_limit_voltage_stratum_kv_ll = 69.0`. The band is
opt-in like the reviewed profile's `weak` band: it is **not** a member of
`bands_ohm`, so the curriculum weights (2:2:1 early, 1:1:1 full) and
`hif_resistance_ohm(band=None)` draws are unchanged and never reach 1000 ohm.
Request it with `hif_resistance_ohm(rng, band="detection_limit")` or the exact
bounds `[1000.0, 5000.0]`; integer indices remain 0/1/2.

`physical_hif_slots` adds two slots, `hif_69kv_detection_limit_0` and
`hif_69kv_detection_limit_1`, for `split != "train"` only, with options
`{"hif_ohm_band": "detection_limit", "voltage_kv": 69.0, "destination":
"hif_detection_limit_evaluation"}`. Rows are written to
`hif_detection_limit_evaluation_manifest.jsonl` with `training_eligible=false`,
`hif_population="69kv_detection_limit_1000_to_5000_ohm"` and
`hif_resistance_class="extreme_weak_hif"`; they are retained independently of
WLS detection and paired-separation admission. The persist train guard rejects
the cohort in `split="train"` a second time. The recorded sweep above shows
0/42 baseline alarms at 1000, 2000 and 5000 ohm on 69 kV, so these rows are
boundary evidence, not training proposals. The cohort is 69 kV only; the
13.8 kV stratum stays covered by the discrete sweep (5000 ohm there is 2625 pu
and produced zero newly alarming windows). The generation report gains
`profiles.hif_detection_limit` and `profiles.hif_resistance_classification`.

Under the physical profile the mixed `measurement_hif` slot no longer carries
the legacy pu `resistance_range` option (dead metadata there, because mixed
slots reuse an accepted physical HIF component). The shared `SLOTS` tuple and
the reviewed profile are unchanged.

### Injector default-resistance rule

`three_phase_model.disturbances.inject_midspan_hif` called with neither
`resistance_ohm` nor `resistance_pu` keeps the historical 10 pu default **only**
when every registry bus shares one `kv_ll` (the legacy uniform normalized
model; e.g. case57 or the default case14 export at 1 kV, where 10 pu is 0.1
model-ohm). On a multi-voltage registry such as the physical IEEE-14 export it
raises `ValueError("resistance_ohm or resistance_pu is required for a
multi-voltage registry")` before any circuit mutation, because the default
would silently mean 476 ohm on a 69 kV line but 19.04 ohm on a 13.8 kV line.
No production path relied on the default: the practical corpus, the sweep
audit and observable-context replay always pass a unit.

Recorded runtime for reference: the 672-case default sweep took 121.03 s
(`output/ieee14_physical_hif_20260918/sweep/summary.json`, single BLAS thread);
the four-case smoke run took 1.07 s.
