**IEEE 57 measurement and parameter scenarios are implemented**

Subsequent threshold work and fresh validation are recorded in [IEEE57 dual detection](ieee57_dual_detection_20260910.md). The original unchanged-expert results below remain the historical baseline.

The existing scenario/environment/expert pipeline now accepts a canonical IEEE 57 system for clean, measurement, multiple-measurement, parameter, and measurement-plus-parameter scenarios. This stage does not generate detailed topology, harmonic or three-phase fault scenarios, and does not train or evaluate an LLM.

The [system registry](../psse_env/systems/registry.py) pins [case57.m](../mcp_server/case57.m) to PYPOWER 5.1.19, with its original bus, generator, branch and OPF cost matrices. It records distinct identities for all 80 branch rows, including parallel circuits, and exposes the 491-channel observation layout and current fixed covariance. Unsupported systems, families and covariance configurations fail explicitly.

The [balanced corpus builder](../psse_env/providers/balanced_corpus.py) generates fresh AC-OPF operating windows and validates telemetry against generator-minus-load injections and stored branch flows, plus voltage/generator bounds. Parameter measurements reflect the changed physical R/X values while the agent starts from the canonical stale case. Repeated scans share one operating point with independent noise; solved true initial states are not supplied to the estimator. Truth cases, labels, physical validation and rejection receipts stay in offline artifacts.

The [existing round-0 generator](../psse_env/providers/scenario_generator.py) now derives dimensions and case selection from the registry. Fresh sources must match case hash, system identity, covariance, vector/scan lengths and declared parameter-only model changes. The default legacy IEEE 14 behavior is preserved. Two admission modes distinguish different uses:

- `physical`: development population from physically validated sources; no WLS detectability or teacher-success selection. Clean false alarms, undetected faults and teacher failures remain measurable.
- `recoverable`: the existing detectability, correction-realizability and teacher-ranking admission checks. Rejections are reported; requested counts can be unmet. This does not replace the later supervision audit.

The source stage always retains its unfiltered clean noise draws. The ten-sigma minimum is applied by the scenario stage to injected meter errors; it does not imply detectability on the larger network.

**Fresh local pilot**

Verified generation manifest (local artifact: `output/ieee57_balanced_development_20260910_verified/manifest.json`)
and scenario envelopes (local artifact: `output/ieee57_balanced_development_20260910_verified/scenarios.json`).

Seed 20260910; load scaling 0.80-1.00; three iid scans per parameter window; fixed sigma(Vm)=0.001 and sigma(power)=0.01. The raw corpus admitted 50/50 attempted OPF windows: 20 clean, 20 measurement-error and 10 parameter-error. No statistical or teacher selection occurred at raw-source admission. This small run does not establish population feasibility or false-alarm calibration.

The development suite contains five roots per family, 25 exact physical roots and 22 parent operating realizations. Three selected parameter/mixed pairs share a source window. Source-window IDs survive in offline envelope grouping; keep them together before any later training/validation/test split. This artifact is development-only and is not a sealed final test.

The unchanged-expert evaluation (local artifact: `output/ieee57_balanced_development_20260910/expert_validation.json`) ran all 25 roots with the existing strict production environment, 40-step limit and existing truth audits. Both generated scenario files are byte-identical, SHA-256 `5c0f69d7d3d345a7e4aa9f56ad5d3ea58e3962eb3fd88b6cd5462d93cfc7f568`; the evaluation therefore binds to the verified suite as well. The second generation records the tightened source-validation implementation and reproduces the original scenario bytes; it is not a second expert evaluation.

| Family | Roots | Existing truth-audited task success | Initially detected anomalies |
|---|---:|---:|---:|
| Clean | 5 | 5/5 | 0/5 |
| Measurement | 5 | 4/5 | 4/5 |
| Multiple measurements | 5 | 4/5 | 5/5 |
| Parameter | 5 | 3/5 | 3/5 |
| Measurement + parameter | 5 | 1/5 | 5/5 |
| Total | 25 | 17/25 | 17/25 |

**Scoring qualification:** the task metric accepts an independently audited controller handoff. Successful corrected fault cases ended in that handoff, not a runtime `resolved` terminal. Only the five clean controls meet the stricter `final_physical_success` field. The complete report retains actual terminal outcomes and the separate counterfactual completion audits; 17/25 must not be called 17 fully autonomous resolved episodes.

The eight task failures are retained: one single-meter and two parameter faults were undetected and falsely finalized; one multi-meter case and four mixed cases had incomplete repairs. There were three false finalizations, zero false commits and zero invalid actions. Healthy components were preserved on all 25 roots, task-outcome evidence was known for all 25, and no infrastructure errors occurred.

Actual pilot parameter successes corrected zero-based branch rows 69, 27 and 24, demonstrating execution beyond IEEE 14's branch range. All 34 harmonic/three-phase acquisition requests returned `unavailable`; no dummy telemetry was supplied. The unchanged protocol still requires these acquisition attempts where applicable. Unavailable is unmeasured, not a negative physical diagnosis.

**Validation completed**

141 distinct targeted tests passed across system registry (6), fresh corpus (8), new system/scenario integration (7), real IEEE 57 expert integration (5), existing generator partition/admission/operating-point tests (30), and existing evaluator/suite/case-loader tests (85). The real tool test separately recovered meter index 450 and parameter line index 22. The line-22 unit test deliberately uses varied scans as a solver-capability check; the actual development corpus uses the iid-window contract described above.

The stricter `recoverable` mode also built one root in each of the five families from the fresh corpus, recording 15 rejected attempts for insufficient detectability. This curated smoke population is separate from the unfiltered 25-root development result. Recoverable-mode receipt (local artifact: `output/ieee57_balanced_development_20260910_verified/recoverable_smoke_report.json`) and combined verification receipt (local artifact: `output/ieee57_balanced_development_20260910_verified/verification_receipt.json`).

A comparison against the scenario-generator file saved before this implementation produced exactly identical serialized IEEE 14 scenarios for one root in each of the five balanced families. Compatibility receipt (local artifact: `tmp/ieee57_pre_edit_hgqp1rc4/compatibility.json`). This is a scoped compatibility check, not a rerun of the historical 160-root policy experiment.

The canonical repository IEEE 14 case has no thermal flow constraints. Its PYPOWER OPF path exposed two upstream empty-array shape errors. The shared OPF helper now scopes and restores compatibility hooks for those empty arrays, preserving all original ratings and numerical constraints. Tests cover restoration on exception and agreement with a distant nonbinding reference. IEEE 57 uses the ordinary solver path.

**Reproduce from the repository root**

Generate a new development population; choose a new output directory because the command refuses to overwrite an existing run:

```powershell
python scripts/build_balanced_transfer_scenarios.py --system case57 --output-dir output/ieee57_next_run --per-family 5 --num-scans 3 --admission-mode physical
```

Run the unchanged expert and preserve complete traces and truth audits:

```powershell
python scripts/validate_balanced_transfer.py --scenarios output/ieee57_next_run/scenarios.json --output output/ieee57_next_run/expert_validation.json
```

The builder returns exit code 2 for an incomplete family plan after preserving its artifacts. The expert validator returns exit code 2 for missing/infrastructure evidence; known teacher failures are reported as outcomes and do not discard the run. The canonical envelopes can be consumed by the existing research workflow.

The implemented milestone is fresh balanced scenario generation and local tool/teacher validation. The next experiment is frozen IEEE 14 policy evaluation on IEEE 57 development data, keeping the current teacher failures visible. Target-system fine-tuning, larger coverage studies, and sealed final-test design remain separate subsequent work.
