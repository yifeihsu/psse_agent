# IEEE57 three-phase availability and physical-ohm HIF detection

The repository already provides a registry-driven IEEE57 three-phase model,
with 57 external buses, 171 phase nodes, 80 branch assets, 63 lines and 17
transformers. The external balanced WLS measurement vector has 491 entries;
its 113 estimated states leave 378 residual degrees of freedom.

The selected physical reconstruction assigns buses 1–17 to 138 kV and buses
18–57 to 69 kV, on the 100 MVA three-phase base. This convention is stated in
the supplied [WSEAS implementation](https://wseas.com/journals/ps/2023/a785116-025%282023%29.pdf).
It is a named reconstruction, `ieee57_reconstruction_138_69kv_v1`, rather than
voltage metadata inferred from the canonical source. The local `case57.m`
actually has BASE_KV=0 throughout; the supplied PGLib variant's unit placeholders
likewise do not establish physical equipment voltages. The canonical case is
preserved, and per-unit R/X/charging/taps and power data are unchanged.

## Available model and generation

A fresh built, independently validated physical model is at
`output/ieee57_physical_hif_20260919/model/Master.dss`. Its balanced model,
physical load-unbalance example and restoration checks all passed.

```powershell
python scripts/build_three_phase_model.py --system case57 --voltage-profile ieee57_reconstruction_138_69kv_v1 --output-dir output/new_ieee57_physical_model
```

The model includes explicit sequence and grounding assumptions, grounded-wye
transformers, and fixed-PQ generator snapshots. It is a fundamental-frequency
research realization. Nonlinear arcing, dynamic generator/AVR response and
field-equipment calibration are not established by these checks. The existence
of this model does not certify an IEEE57 HIF DAgger rollout or a trained policy.

There are 26 eligible same-voltage lines at 138 kV and 37 at 69 kV. Transformer
assets, including nominal-tap and same-voltage regulating transformers, are
excluded from midspan HIF injection.

## Ohmic severity

Faults are specified in physical ohms and converted with the faulted line's
local base, `Zbase = kV_LL**2 / MVA_3ph`:

| Rfault | Rpu at 138 kV, Zbase=190.44 ohm | Rpu at 69 kV, Zbase=47.61 ohm |
| --- | ---: | ---: |
| 50 ohm | 0.2625 | 1.0502 |
| 100 ohm | 0.5251 | 2.1004 |
| 200 ohm | 1.0502 | 4.2008 |
| 500 ohm | 2.6255 | 10.5020 |
| 1000 ohm | 5.2510 | 21.0040 |
| 2000 ohm | 10.5020 | 42.0080 |
| 5000 ohm | 26.2550 | 105.0200 |

Thus the same pu resistance corresponds to four times as many ohms at 138 kV.
The sweep records actual solved fault current, voltage, power and both resistance
units; nominal `Vphase/R` estimates do not replace the circuit solution.

## Fresh experiment design

The reproducible sweep entry point uses disjoint line shards, with one native
BLAS thread per worker. A fresh output directory is required:

```powershell
python scripts/audit_ieee57_hif_physical_sweep.py --output-dir output/new_ieee57_physical_sweep --workers 8
```

The physicalized sweep uses the same resistances on every eligible line and
phase, with alpha=0.5 and two load multipliers, 0.8 and 1.0. This is 2,646
physical fault cases, not 2,646 independent operating parents. Each case is
re-solved independently in OpenDSS, checked for circuit consistency and
constant-PQ behavior, and compared with an equivalent local-pu injection.
Paired healthy and resistor-disabled split controls preserve external sensor
identities and the original line charging.

Actual Gaussian noise and WLS covariance match: sigma(Vm)=0.001 pu, and
sigma(P/Q)=0.01, 0.005 or 0.002 pu in three separately reported profiles.
There are 378 independent standardized-noise groups, shared across resistance,
accuracy and paired controls. Noisy fault/control observation counts must not
be presented as independent operating-point counts.

The primary detector matches `psse_env/dagger/ieee57_runtime.py`:

`J >= chi2.ppf(0.95, 378) OR max(abs(normalized_residual)) >= 4`.

The global threshold is **424.334166**. A separately labelled 1% comparison
uses **444.888945**, retaining the same NR threshold, observations and fitted
J/NR. It does not resimulate or change the noise. This separates the detector
setting from the network/voltage comparison with IEEE14.

Quiet physical cases and failed fits remain explicit. A quiet WLS result is not
proof that no HIF exists. Weak-case alarms must be compared with their matched
healthy-noise alarms before being credited as additional fault evidence.

## Completed 2026-09-19 physical-ohm results

The full eight-worker sweep completed in 731.94 seconds. All **2,646 physical
HIF cases**, **7,938 fault WLS observations** and **2,268 paired control WLS
observations** are present. There were zero physical failures, zero failed
fault/control WLS fits and no changes to the recorded implementation during
collection. The raw collection and its source snapshots are preserved at
[`sweep/summary.json`](../output/ieee57_physical_hif_20260919/sweep/summary.json).

The following percentages are the observed dual-alarm rates at the primary
5% chi-square setting. Each 138 kV cell has 156 cases (26 lines x 3 phases x 2
parents); each 69 kV cell has 222 cases (37 x 3 x 2). Voltage noise is fixed at
0.001 pu. Power-noise standard deviations are absolute pu on the 100 MVA base,
not percentages of each channel's value.

| Rfault (ohm) | 138 kV, sigmaPQ=.01 | 69 kV, sigmaPQ=.01 | 138 kV, sigmaPQ=.005 | 69 kV, sigmaPQ=.005 | 138 kV, sigmaPQ=.002 | 69 kV, sigmaPQ=.002 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 100% | 100% | 100% | 100% | 100% | 100% |
| 100 | 100% | 100% | 100% | 100% | 100% | 100% |
| 200 | 100% | 99.1% | 100% | 100% | 100% | 100% |
| 500 | 100% | 28.4% | 100% | 83.8% | 100% | 100% |
| 1000 | 59.6% | 14.4% | 100% | 20.7% | 100% | 96.4% |
| 2000 | 13.5% | 9.9% | 58.3% | 12.6% | 100% | 27.0% |
| 5000 | 6.4% | 9.5% | 7.7% | 8.6% | 57.7% | 10.4% |

The baseline healthy-noise false-alarm rate was **32/378 (8.47%)**, comprising
18 chi-square alarms and 14 additional NR-only alarms. The 0.005 and 0.002
power-noise profiles each produced 29/378 healthy alarms (7.67%). Healthy
observations use the entire 491-channel system; their voltage labels identify
the faulted lines used for pairing, not separate voltage-only detectors.

At chi-square alpha=0.01 with the same NR>=4 rule and identical fitted
observations, baseline healthy alarms fell to **18/378 (4.76%)**. Tightening
the global test alone does not calibrate the combined OR detector to 1%.
The full alternative results are in
[`detection_by_voltage_resistance_alpha_0p01.csv`](../output/ieee57_physical_hif_20260919/sweep/detection_by_voltage_resistance_alpha_0p01.csv).

The independent [pairing audit](../output/ieee57_physical_hif_20260919/sweep/independent_pairing_audit/audit.json)
checked the full Cartesian population, reconstructed all 10,206 observation
hashes from physical means plus declared noise, verified all 491-element
covariance vectors and both detector gates, and confirmed identical alarm
decisions for healthy versus no-fault split controls. At baseline noise and
5,000 ohm, **neither voltage group produced a new alarm**: all 10/156 alarms
at 138 kV and 21/222 alarms at 69 kV were already present in the paired healthy
observations. At 69 kV and 2,000 ohm, only 2 of the 22 alarms were new.
The [paired table](../output/ieee57_physical_hif_20260919/sweep/independent_pairing_audit/paired_detection.csv)
reports new, shared, cleared and both-quiet outcomes separately, including
rates conditional on the healthy observation being quiet. That conditional
denominator changes when the threshold changes.

Noiseless healthy J was at most 3.82e-11, while noisy healthy J reached 466.23.
The near-zero noiseless result is the expected deterministic model agreement;
it is not a post-repair noisy-episode result. Actual random measurement noise
was retained for every reported fault/control alarm rate.

These measurements establish strong baseline observability through 500 ohm
on the tested 138 kV lines and approximately 200 ohm on the tested 69 kV
lines. Better matched measurement accuracy extends this range, but very weak
fault alarms must be assessed against paired healthy controls. These are
synthetic coverage results under the diagonal sequence/grounding completion,
two operating parents and 378 standardized-noise draws, not field detection
probabilities or a coupled-sequence sensitivity study.

Balanced, load-unbalance and restoration checks passed for the public model.
The focused exporter, voltage-base, WLS-invariance and sweep regression suite
passed 108 tests; its [JUnit record](../output/ieee57_physical_hif_20260919/regression.xml)
is preserved. This validates the standalone model and detection experiment,
not HIF localization, physical recovery, learned-policy performance or a
completed IEEE57 DAgger training corpus. For a WLS-gated training corpus,
fault-attributable evidence and the subsequent expert diagnostic action still
need to be checked; a raw weak-fault alarm alone is insufficient admission
evidence.

The merger now rejects incomplete case/control/noise-profile combinations,
including missing noisy observations. Its 24 focused tests passed after this
coverage guard was added. The [separate post-run coverage receipt](../output/ieee57_physical_hif_20260919/sweep/post_run_coverage_validation.json)
applies that stricter check to the saved complete population and records the
original collection code and later validator hashes separately; the original
collection files and snapshots were not rewritten.

## Historical results are different evidence

The earlier verified study at
`output/ieee57_hif_unbalance_20260911_verified/` used a uniform 1 kV normalized
realization and resistances 10/100/1000 pu. Its actual noisy WLS results were:

| Resistance | WLS alarms |
| --- | ---: |
| 10 pu | 123/252 (48.8%) |
| 100 pu | 22/252 (8.7%) |
| 1000 pu | 14/252 (5.6%) |
| Total | 159/756 (21.0%) |

Healthy controls produced 42/400 alarms (10.5%): 24 chi-square and 18 additional
NR-only alarms. Those noise realizations share four physical roots. The
476/756 HIF result in the same study is **phase-localization accuracy**, not WLS
detection. All these counts are historical artifact observations, not a rerun
under the new voltage reconstruction or proof of current learned-agent success.

The new voltage/ohm experiment must therefore be reported separately. Neither
study's resistance sweeps are a calibrated distribution of field HIF events.
