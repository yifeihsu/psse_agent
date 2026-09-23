# Healthy-control auditing and HIF-conditioned meter repair

Implemented in the main `PSSE_Agent` checkout on `codex/wls-screen-gnn`, based on
`1cf17e63c7de161304f87afb6684b351e6488ee4`. This change uses the existing agent
actions and the physical HIF replay helper. It does not physically clear an HIF.

## Audit reference

A complete, zero-fault healthy control is compared with its original noisy
acquisition. Ordinary sensor noise is not an error that the agent must remove.
Any healthy-channel change still fails preservation; the noiseless reference
remains available as separate experimental truth.

For a mixed HIF/measurement scenario, the measurement reference is the original
HIF-present acquisition with only the declared gross-error targets restored to
their recorded pre-overlay values. The final-vector and accepted-target audits
use the same reference. Healthy channels must remain exactly unchanged. The
HIF diagnosis, meter-target correctness and remaining-fault checks still apply.
An inherited pure-HIF measurement waiver is replaced by this full comparison,
never used to excuse an unrepaired meter. Incomplete or malformed truth fails
closed. Reference selection is offline audit logic, unavailable to the policy.

## Runtime method

After any accepted HIF fit, including a pure-HIF episode, the controller requires
a new, state-bound residual check. It does not use scenario-family truth to
decide whether another fault might be present.

The current acquisition is identified by its auxiliary phase-voltage and
branch-current measurements. Its SCADA scan is excluded from multiscan fitting.
The remaining observable scans estimate the HIF parameters. Paired OpenDSS
solves at the current observable operating point predict the HIF-present and
HIF-absent external measurement vectors. The temporary WLS input is

`z_conditioned = z_observed - (z_predicted_HIF - z_predicted_no_HIF)`.

All channels remain present, with the original sensor covariance. This
subtracts the estimated physical contribution, rather than zeroing residuals
or discarding affected meters. A coexisting meter error remains in this vector,
including when it occupies the same channel as the HIF contribution.

The provider additionally checks discrepancy from the HIF-present prediction.
Candidates must exceed 5 measurement sigmas outside a sampled sensitivity
range. That range uses the best fit and corners of its near-best alpha/resistance
rectangle; its full width must not exceed 2 sigmas. Broad discrepancy patterns,
missing operating-point/acquisition evidence, stale models, unsupported
conventions and unavailable local residual tests require an operator handoff.

Only currently supported meter targets can be corrected. Their replacement is
the HIF-present predicted value; every other acquired reading is retained
exactly. Candidate verification repeats compensation on the new measurements.
Case, auxiliary acquisition and DSS model fingerprints prevent stale fit reuse.
The supported runtime convention is `ybus`; legacy injection conventions need
a separately adapted observation model and are not silently accepted.

The sequence is HIF localization and fitting, conditioned WLS, meter context,
meter correction, candidate WLS, commit, and the existing operator handoff after
correction. A premature `finalize_diagnosis` is invalid. Diagnosis plus meter
repair does not imply HIF removal or restoration of voltage operating limits.

A narrowly scoped partial-acceptance check permits meter repairs when the only
remaining physical violations are pre-existing, unchanged voltage readings.
It independently requires unchanged network data, exact preservation of
non-target meters and all voltage channels, quiet conditioned diagnostics,
connected topology, complete evidence, and no thermal violation. The original
`physical_constraints_ok=false` is retained. This route permits only partial
acceptance followed by operator review; changed voltage violations, thermal or
topology problems and unknown physical status cannot use it.

## Evidence and limitations

The completed `output/hif_continuation_fix_20260922/final_verified/` run uses the
final code and immutable, model-bound fresh-fit cache. Its source and DSS model
stability checks both passed. Earlier unsuccessful integration attempts are
retained in separate directories and excluded from these results.

| Check | Result | Endpoint |
| --- | --- | --- |
| Archived healthy telemetry controls, re-audited unchanged | 8/8 pass | Original noise preserved |
| Original frozen R1 HIF + measurement roots | 8/8 meter recovery and truth-audited task success | Nine actions each; commit then operator handoff |
| Additional 10-sigma errors on the strongest HIF-affected channel | 8/8 detected; 6/8 committed and task-successful | Two explicit failures retained after rollback and handoff |
| Paired HIF-only controls | 8/8 correct diagnostic closure | Four actions each; no meter writes |

There were zero non-target writes across the 24 new episodes. On the eight
original mixed roots, conditioned WLS J fell from 246.067–847.706 before meter
repair to 77.196–114.124 after repair, while the maximum normalized residual
fell from 12.237–27.464 to 2.238–2.935. The largest repaired-meter deviation from
its HIF-present pre-overlay noisy value was 0.014301 pu, below the existing
0.0424264 pu recovery tolerance. Noise was neither redrawn nor removed from
healthy channels.

All eight additional overlap targets had material predicted HIF contributions
(9.57–20.84 measurement sigmas), and every added error remained in the
compensated vector. The unsupported stress cases are
`r0_de51c28ced3e_controlled_overlap` (Vm index 3, bus 4) and
`r0_b32f59dc6fa0_controlled_overlap` (Vm index 2, bus 3). Their meter errors were
detected, but the proposed corrections changed voltage readings that remained
outside operating limits. The unchanged-voltage nonregression exception does
not apply, so both candidates were rolled back and both episodes remain task
failures. Quiet candidate WLS results are not reported as committed recovery.

Machine-readable evidence: [episode table and metrics](../output/hif_continuation_fix_20260922/final_verified/report_metrics.json),
[case table](../output/hif_continuation_fix_20260922/final_verified/report_metrics.csv),
and [healthy-control audit receipt](../output/hif_continuation_fix_20260922/measurement_reference_audit_receipt.json).
Automated checks passed: 566 tests and 216 subtests in the consolidated suites;
after the final controller/wrapper alignment, 128 tests and 91 subtests passed
again, followed by all 28 focused controller tests including certificate
stripping. These counts overlap and must not be summed. Commands and results
are in the [test receipt](../output/hif_continuation_fix_20260922/test_receipt.json).

The regression inputs are the eight exact mixed-HIF development roots from the
completed R1 evaluation, retrieved without changing their source artifacts.
The paired control and added-overlap experiments reuse these roots and are not
independent new physical events. Historical R1 student scores are not overwritten;
new expert/environment traces do not constitute a rerun of the trained student.

The sampled prediction range is not a guaranteed nonlinear bound or confidence
interval. Compensation uncertainty is not added to WLS covariance, so nominal
chi-square false-alarm calibration is not established. Trusted operating-point
context and the matching OpenDSS physical model are required. Results on these
roots do not establish robustness to unknown operating points, auxiliary-sensor
faults, different physical faults or persistent corruption of fitting history.

Reproduction (new output directory; the cache is bound to observable fit inputs):

```powershell
python scripts/verify_hif_continuation.py --frozen-scenarios output/hif_continuation_fix_20260922/frozen_mixed_scenarios.json --output-dir output/hif_continuation_fix_20260922/reproduced --fit-cache-dir output/hif_continuation_fix_20260922/fresh_fit_cache --alpha-grid-size 7 --r-grid-size 9 --max-scans 10 --normalized-residual-threshold 4 --max-steps 40 --controlled-overlap --paired-hif-controls
```

The recorded R1 research settings are chi-square alpha 0.01, normalized-residual
threshold 4, a 7-by-9 HIF grid, up to 10 scans and 40 actions. Holding out the
current scan leaves nine fitting scans on these roots. The frozen measurement
noise and covariance are preserved.
