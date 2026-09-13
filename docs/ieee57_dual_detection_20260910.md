# IEEE57 chi-square and normalized-residual detection

The balanced-transfer validator now uses `chi2_alpha=0.05` and a separate maximum absolute normalized-residual alarm at **4.0**. An anomaly remains if **either** test fails. For the 491-channel IEEE57 vector with 113 estimated states, the chi-square cutoff changes from **444.888945** to **424.334166** (378 degrees of freedom).

```
alarm = (J >= chi2_cutoff) OR (max(abs(normalized_residuals)) >= 4.0)
clean = NOT alarm
remaining_anomaly_score = max(J / chi2_cutoff, max_abs_normalized_residual / 4.0)
```

The local alarm cutoff is separate from the existing 3-sigma candidate-screening and target-repair criteria. Four sigma was chosen before generating the independent clean cohort. A raw three-sigma maximum is too sensitive across 491 channels: it would flag four of the original five clean controls. The global test's nominal 5% level is not a 5% combined false-alarm guarantee. The independent cohort below measures the practical tradeoff; its results were not used to retune the thresholds.

## Implementation

- `psse_env/providers/matpower.py` applies both tests to initial and candidate WLS solves, emits the exact configuration and separate alarm flags, and generates routing evidence for residual-only alarms. Post-measurement branch screening and refinement also use both tests. The WLS summary now uses the configured chi-square threshold instead of a separate implicit default.
- `psse_env/transactional_env.py` retains the residual alarm in the current-state WLS evidence ledger and prevents it from being cleared by the postcorrection confirmation shortcut. Candidate progress remains the relative reduction in **J**, rather than the change in the maximum of two alarm ratios, so existing partial-repair criteria keep their numerical meaning.
- The candidate-quality, process-validity, termination and expert rules veto statistical completion or confirmation handoff while an explicit residual alarm remains. Existing candidate safety and partial-progress floors are preserved.
- Evaluator, study-certificate and policy-summary paths preserve the new evidence. Certificates check the conjunction of the configured tests. Older certificates without the optional residual criterion retain their original interpretation.
- `scripts/validate_balanced_transfer.py` exposes both thresholds and records their effective values. The general production factory retains its historical defaults; this balanced-transfer experiment explicitly enables the new criteria. No learned policy was trained or evaluated.

## Same 25 physical scenarios, fresh execution

[Full audit](../output/ieee57_dual_threshold_20260910/expert_validation.json) and [comparison](../output/ieee57_dual_threshold_20260910/baseline_comparison.json).

| Family | Previous audited task successes | New audited task successes |
|---|---:|---:|
| Clean | 5/5 | 5/5 |
| Measurement | 4/5 | 4/5 |
| Multiple measurements | 4/5 | 5/5 |
| Parameter | 3/5 | 4/5 |
| Measurement + parameter | 1/5 | 3/5 |
| Total | **17/25** | **21/25** |

Initial fault alarms increase from 17/20 to 18/20. False finalizations decrease from three to two. The 25-root pilot has zero recorded false commits, zero invalid actions, and healthy components preserved on all 25 roots. There are no infrastructure errors or task-outcome regressions.

The multiple-meter case now corrects both indices 90 and 97. The previously ignored 9.38 normalized residual remains active through confirmation. Two mixed cases now continue to repair the remaining branch parameter, and one parameter-only case is newly detected through its residual alarm.

The four remaining failures are retained:

| Scenario | Remaining fault | J when diagnosis stops | Maximum normalized residual |
|---|---|---:|---:|
| `r0_0d5add3d6a4c` | Meter 19 | 406.744 | 3.974 |
| `r0_76f8b1fde026` | Branch row 22 parameter | 408.218 | 3.310 |
| `r0_389b463fd4c9` | Branch row 22 parameter after meter repair | 405.644 | 3.310 |
| `r0_684f4164e4dd` | Branch row 32 parameter after meter repair | 295.758 | 2.539 |

These observations pass both new tests, so threshold changes alone do not establish complete fault coverage. Meter and branch rows above are zero-based.

The task metric includes independently audited controller handoffs. The new run still has only **five strict runtime-resolved physical successes**, all clean controls. The 21/25 task result must not be described as 21 autonomous resolved episodes.

The scenario input SHA-256 is unchanged: `5c0f69d7d3d345a7e4aa9f56ad5d3ea58e3962eb3fd88b6cd5462d93cfc7f568`. All 21 selected source files had matching hashes immediately before and after the fresh execution. The old audit remains unchanged.

## Independent clean operating windows

After fixing the thresholds, generated **100 new physically validated clean windows**, seed 20260911, load scale 0.80–1.00, fixed measurement covariance. All 100 were admitted without WLS or expert-success filtering. The independent corpus is separate from the 25-root pilot.

[Initial detection audit](../output/ieee57_dual_threshold_20260910/independent_clean_detection.json), [full expert trajectories](../output/ieee57_dual_threshold_20260910/independent_clean_expert_validation.json), and [compact findings](../output/ieee57_dual_threshold_20260910/independent_clean_summary.json).

- Historical criteria flag 0/100 at initial detection; the new criteria flag **6/100**.
- All six trigger the lower chi-square cutoff; one also exceeds the 4-sigma residual cutoff.
- The new expert has **97/100 audited task successes** and preserves healthy components on 97/100 windows. Ninety-four finalize directly, three escalate without changing a healthy component, and three incorrectly correct a healthy meter before handing off.
- The incorrect corrections affect meter 389 in `r0_04b79fcafe53`, meter 291 in `r0_6b96d3601e5f`, and meter 354 in `r0_9502fe57389e`. The strict target and healthy-preservation audits reject all three.

The legacy `false_commit_count` field reports zero even in these three failed clean trajectories. It therefore does not establish absence of incorrect corrections; the explicit accepted-target and healthy-preservation audits are the decisive evidence here. The threshold change improves the pilot's fault recovery, but stronger correction evidence is still needed to prevent these clean-window errors. No thresholds were tuned again after seeing this cohort.

## Regression checks and reproduction

Passed 6 dual-criterion provider tests, 61 existing provider tests, 8 balanced-transfer integration tests, 214 oracle/production tests, and 139 evaluator/study/export tests. An additional overlapping 95-test postcorrection regression run also passed. Checks cover threshold equality, residual-only anomalies, partial-repair progress, finalization and handoff vetoes, legacy configuration, nonfinite evidence, and certificate consistency.

```powershell
python scripts/validate_balanced_transfer.py --scenarios output/ieee57_balanced_development_20260910_verified/scenarios.json --output output/ieee57_next_dual_audit.json --chi2-alpha 0.05 --normalized-residual-threshold 4.0
```

To explicitly replay the old detection settings with current code, use `--chi2-alpha 0.01 --chi-square-only`. This selects the old criteria; it does not replace the preserved historical runtime artifact. The independent clean generation and rollout scripts are saved beside their output artifacts.
