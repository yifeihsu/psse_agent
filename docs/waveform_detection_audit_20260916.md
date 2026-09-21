# HIF and unbalance detection at the current thresholds

Fresh audit in the PSSE_Agent repository, 2026-09-16, current base HEAD
`a6550caa70ce40515785e03241c9307c6317eafb` plus the local noise-alignment changes.
No detector threshold or production detector implementation was changed in this
audit. Source data and evaluation outputs are under
`output/waveform_detection_20260916/`.

**Conclusion:** HIF is detectable through the configured multiscan current
diagnostic on all 17 tested roots. Single-snapshot HIF detection is weaker.
The current-assisted unbalance detector has high sensitivity but is not reliable
with the aligned voltage noise: it falsely accepts 123/200 healthy controls.
Balanced WLS also has an unresolved measurement-convention issue on these
OpenDSS observations, so its apparent perfect fault detection is misleading.

**Inputs and settings**

The primary fault inputs are all 17 development HIF windows and all 220
unbalance source rows, prepared with matched sensor noise. No roots were
discarded based on their WLS or auxiliary detector result. Source labels and
clean references are used only after decisions for audit. HIF WLS uses scan 0;
the multiscan detector uses the window's independently noisy scans.

The 200 healthy controls use 50 distinct operating points and four independent
noise replicates each, drawn from 30 original source-event groups. Real balanced
OpenDSS means were solved without a HIF or unbalance injection. Each control has
fresh independent SCADA, phase-voltage and branch-current noise; no WLS or
diagnostic admission filter was applied.

The research pipeline settings are:

- WLS global alpha 0.01, nominal dof 95, threshold **129.9726787**.
- Maximum normalized residual threshold **4.0**.
- WLS alarm when either threshold is met or exceeded.
- HIF current differential: **6 sigma**; single-snapshot threshold
  `6*sqrt(2)*sigma_I = 0.00848528 pu` at sigma_I=0.001.
- Coherent HIF multiscan threshold divides by sqrt(N): **0.003 pu for eight
  scans**, **0.00268328 pu for ten scans**.
- Voltage-only unbalance: maximum VUF >= **1%**.
- Current-assisted unbalance: significant source phase-power spread and a
  quiet HIF line-differential test. With currents present this replaces, rather
  than supplements, the VUF gate.

The source-spread significance floor is currently
`6 * abs(V1) * sigma_I * sqrt(incident branch count)`. Relative source ranking
uses a 0.02 pu mean-power floor. These are current code settings, not retuned
values selected after observing the results.

**Specialized detector results after telemetry acquisition**

| Test | Fault result | Healthy-control false alarms |
|---|---:|---:|
| Single-snapshot HIF differential | 10/17 detected | 0/200 |
| HIF differential using all ten scans | 17/17 detected; correct branch and phase | Multiscan false alarms not evaluated in the 200 snapshot controls |
| HIF using Round0's eight-scan selection | 17/17 detected; correct branch and phase | Same qualification |
| Unbalance VUF alone | 143/220 detected | 10/200 (5%) |
| Current-assisted unbalance acceptance | 217/220 accepted | 123/200 (61.5%) |
| Accepted unbalance with correct source bus | 213/220 | Source accuracy is not defined on healthy inputs |

No primary fault input was missing and there were no diagnostic exceptions.
The seven single-snapshot HIF misses still rank the correct branch/phase first,
but their score/threshold ratios are 0.549–0.980, below the alarm threshold.
Detection is distinct from accepting a fitted HIF model or recovering its
along-line position and resistance. No full HIF grid-fit or episode-success
claim is made by these counts.

The single-snapshot unbalance rule also accepts **6/17 HIF roots** when the
weak HIF differential does not reject the line-null hypothesis. Five select
bus 1 and one selects bus 3. Multiscan HIF detection sees all these faults in
this corpus. No unbalance root triggered the HIF differential alarm (0/220).

Three unbalance roots are not accepted. Four additional accepted roots identify
the wrong bus, all choosing bus 1. These remain misses/mislocalizations in the
reported denominator, not successful detections with corrected labels.

**The unbalance false positives come from unpropagated voltage noise**

The detector forms source power using V*conj(I), but its significance floor
accounts for current noise without the voltage-noise contribution. At high
current, voltage uncertainty contributes substantial power uncertainty.

A paired diagnostic ablation preserves each of the exact same noisy current
samples and substitutes its corresponding noiseless phase-voltage mean:

| Healthy-control diagnostic input | Unbalance false acceptance |
|---|---:|
| Properly noisy voltage and current measurements | 123/200 |
| Same noisy currents, noiseless voltage means | 0/200 |

All 123 false acceptances disappear in that ablation. Their original reported
sources were bus 1 in 106 cases and bus 3 in 17 cases. Forty-nine of the 50
operating-point groups had at least one false acceptance. All noiseless parent
means had no significant unbalance source and no HIF line differential.

The ablation is only a mechanism check. It is not a proposed return to noiseless
voltage measurements, and is not counted as aligned-sensor performance. The
actual noisy controls remain unchanged. Their empirical standardized noise SD
is approximately 1.003 for SCADA, 1.003 for voltage components and 1.001 for
current components.

Raw sensor covariance matching therefore did not finish detector validation.
The derived current/source statistics need joint voltage/current uncertainty
propagation and subsequent clean-control validation. Lowering a scalar threshold
or removing noise would not address this defect.

**Initial WLS detection and the OpenDSS measurement convention**

The generic runtime WLS still uses canonical case14 injection semantics. These
OpenDSS observations include capacitor supply in their exported bus injection.
The noise-alignment work forwards covariance but does not adapt this mean
measurement convention. The earlier standalone HIF recovery experiment used
a matching branch-net operator; that adapter is not active in generic runtime.

Both operators were evaluated on the identical observations and covariance:

| WLS input cohort | Current canonical operator: either alarm | Matching branch-net convention: either alarm |
|---|---:|---:|
| HIF roots | 17/17 | 0/17 |
| Unbalance roots | 220/220 | 94/220 (42.7%) |
| Healthy controls | 200/200 | 2/200 (1%) |

Every solve converged. The matching-convention column is a paired diagnostic
check, not a deployed code change. It retains the physical capacitor, original
observations and sensor covariance; only the observation-model convention
changes. Its unbalance alarms comprise 85 global alarms, 86 local alarms,
and 94 in their union. Its two healthy alarms comprise one global and one
local alarm. The current operator raises both alarms on every row in all three
cohorts, including every healthy control.

Thus the current WLS alarms cannot establish meaningful HIF/unbalance detection
on these inputs. After accounting for the convention mismatch, these weak HIF
roots are not visible to balanced WLS at the current thresholds, and more than
half the unbalance roots are also quiet.

**Routing and scope**

HIF defaults to a seeded zero-sequence/relay flag, permitting its specialized
route. Those flagged episodes are not tests of balanced-WLS discovery
sensitivity. Unbalance defaults to discovery, normally requiring a WLS anomaly
before phase acquisition. Quiet WLS can therefore prevent the necessary
diagnostic measurements from being requested. An explicitly configured GNN
phase-investigation trigger can provide another route, but the standard
research factory does not enable it by default.

Before claiming reliable end-to-end detection, the remaining work is to:

1. Bind the generic WLS observation model to the actual sensor convention.
2. Propagate both voltage and current covariance through the derived HIF/null
   and unbalance statistics; validate false alarms without selection filtering.
3. Provide an independent phase/current acquisition trigger for conditions
   that are not observable through balanced SCADA alone.

These rates are preliminary synthetic-corpus evidence. The 200 controls contain
repeated draws from 50 parents; they do not establish a population guarantee.
Auxiliary acceptance after acquisition is not an end-to-end episode false-alarm
rate, and branch/phase detection is not full physical parameter recovery.

**Artifacts**

- `preparation.json`, `prepared_faults.jsonl`: exact shared aligned fault inputs.
- `auxiliary_audit.json`: all auxiliary detections, thresholds and truth audits.
- `healthy_controls/control_summary.json` and `samples.jsonl`: actual noisy
  controls, provider decisions, independent-rule agreement and source grouping.
- `healthy_controls/paired_voltage_ablation_summary.json`: paired cause check.
- `wls_audit.json`: global/local tests separately for both measurement operators.

Re-run WLS comparison with `python scripts/audit_waveform_detection.py` after
preparing the saved input files. Control/auxiliary reproduction scripts and
per-snapshot records accompany their output artifacts.
