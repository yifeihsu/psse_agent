# IEEE57 correction-method applicability and C:\dw continuation

Reviewed 2026-09-12 from `C:\dw`, branch `codex/ieee57-topology-review-20260912`, based on commit `937ac51626c76c539d3d269f7513a4cd7fc0c8d3` plus the copied working changes.

## Decision

The proposed method applies to the IEEE57 logical-CB testbed. Keep discrete topology-hypothesis testing and fixed-evidence AC-WLS as the correction backend. Add normalized-residual branch ranking and normalized constraint-multiplier coupler ranking as diagnostic layers. Verify proposed statuses through the existing numerical and ambiguity checks before applying them.

Most of the correction backend already exists. The current ideal-coupler contraction is an exact reduced-coordinate representation of the proposed equality constraints; it does not need to be replaced with a small-impedance model or the IEEE14 detailed-substation estimator.

This task synchronized the worktree, reviewed applicability, and ran a standalone numerical feasibility experiment. It did not integrate a new ranker, constrained solver, WLAV backend, or learned-agent protocol into production code.

## Worktree synchronization

`C:\dw` was a clean linked Git worktree at detached commit `7b1d9a0`. It now has a separate branch at `937ac51`, followed by an exact copy of the source's 10 tracked working-file changes and 19,461 non-ignored untracked files. The copied data total 7,325,856,872 bytes. Every copied file was checked by SHA-256. The target tracked diff matches the source tracked diff.

The source at `C:\Users\Holiday\Documents\ChatGPT\PSSE_Agent` retained its HEAD, working diff, and status throughout synchronization. Existing target ignored data/caches had no copy collisions. The worktrees share Git objects and refs, but have separate checked-out files and indices. All review commands and new artifacts run under `C:\dw`; nothing was pushed or submitted to HPC.

Sync receipt and file hashes (local artifact: `output/worktree_sync_20260912/sync_receipt.json`). Historical audit files were copied without rewriting their embedded provenance paths. The final historical topology result is the `_verified_v2` directory; earlier `_verified` and `_verified_final` directories remain partial historical attempts.

## Proposal compared with the current implementation

| Proposed component | Current IEEE57 status | Assessment |
|---|---|---|
| One binary asset-status CB per physical branch | Implemented for 80 canonical branches | Direct match. Preserve branch identity, charging, taps, and phase shift. |
| Same raw observations, covariance, and availability for each hypothesis | Implemented | Retain this contract. The full section model has 530 raw channels. |
| Unchanged model plus all single flips | Implemented | 81 hypotheses for branch-only; 94 for all 80 branches and 13 couplers. |
| Residual-based branch candidate ranking | Not implemented as a device ranker | Add mapping from physical residual rows to incident branch candidates; assess shortlist recall. |
| Ideal open/closed coupler constraints | Implemented by exact elimination | Closed: shared voltage/angle plus free P/Q transfer. Open: separate voltage/angle and zero transfer. |
| Coupler NLM screening | Not integrated | Demonstrated numerically in the standalone experiment below. |
| Joint status hypotheses | Optional pair enumeration exists | Add an adaptive expansion/controller policy if needed; keep scope explicit. |
| Genuinely unknown statuses | Binary assignment enumeration exists | Retain ambiguity when both statuses fit; there is no assumed-status NLM to test initially. |
| Explicit, version-bound software status corrections | Implemented | Certificates bind the model, observations, hypothesis scope, and comparison guard. |
| Multiple WLS starting states | Missing | Current solver tries a common flat start once. |
| Joint topology/analog-error repair and generalized WLAV | Missing | A separate extension, especially for accepting partial repairs while other anomalies remain. |
| Independent threshold/ranking calibration | Not established by existing development audit | Needed before performance or false-correction claims. |
| Complete LLM/DAgger integration for raw section sensors | Not complete | Logical-aware provider hooks exist, but the legacy environment factory and analog/parameter correction routes remain guarded. |

Code: [estimation](../logical_topology/estimation.py#L25), [hypothesis scanning](../logical_topology/runtime.py#L318), [certificate application](../logical_topology/runtime.py#L450), and [provider boundaries](../logical_topology/provider.py#L84).

The proposal's whole-asset interpretation is appropriate here: MATPOWER's `makeYbus` multiplies both series admittance and line charging by branch status while retaining tap and phase-shift factors. [MATPOWER source](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html). This abstraction is different from a physical breaker opened at just one end of an energized line.

## Exact coupler formulation and numerical demonstration

With all 13 synthetic substations expanded, IEEE57 has 70 physical voltage nodes. After choosing one angle reference, an expanded estimator has 139 voltage/angle coordinates and 26 coupler-flow coordinates: 165 variables in total. The 13 known coupler statuses impose 26 independent linear equalities, leaving 139 free coordinates.

For a closed coupler, impose theta_A - theta_B = 0 and V_A - V_B = 0. For an open coupler, impose P_CB = Q_CB = 0. The current estimator eliminates those equalities exactly. A diagnostic layer can reconstruct the expanded Jacobian at the same fitted state and recover the operational-constraint multipliers. A separate constrained solver is useful as an equivalence reference, but a wholesale solver replacement is unnecessary.

The standalone probe uses a nullspace basis to enforce Cx = 0 exactly, without pseudo-measurement weights or epsilon regularization. It derives multiplier covariance by propagating the actual analog noise through the constrained linearized estimator and cross-checks against the full exact KKT sensitivity system. It does not reuse IEEE14 parameter multipliers or its earlier regularized covariance calculation.

Seven direct-sensor configurations were tested using three fresh physical PF worlds, one noise draw per world, and truth-free flat initialization. All 13 couplers were ranked; the true faulty device was used only to construct and evaluate the experiment.

| Configuration | Reported-model J | Maximum normalized residual | True faulty coupler rank | Opposite status passes both residual tests |
|---|---:|---:|---:|---|
| Healthy, all couplers closed | 396.5141 | 2.8310 | N/A | No alternative needed |
| Bus 4 split error: true closed, reported open | 2640.8397 | 35.0132 | 1 | Yes |
| Bus 12 split error: true closed, reported open | 22971.2358 | 81.5280 | 1 | Yes |
| Healthy, bus 4 coupler open | 368.8299 | 2.9562 | N/A | No alternative needed |
| Bus 4 merging error: true open, reported closed | 9437.6498 | 50.3811 | 1 | Yes |
| Healthy, bus 12 coupler open | 421.6702 | 3.5562 | N/A | No alternative needed |
| Bus 12 merging error: true open, reported closed | 42202.5252 | 67.0825 | 1 | Yes |

Every expanded fit had constraint rank 26, observable tangent rank 139, and 391 residual degrees of freedom. Compared with the existing reduced estimator:

- Maximum objective difference: 8.41e-10.
- Maximum predicted-measurement difference: 1.61e-7 pu.
- Maximum equality-constraint violation: 1.67e-16.
- Multiplier covariance versus independent exact-KKT propagation: relative difference at most 2.3e-15.
- Reversing constraint-row order at the fixed fitted state changed normalized multipliers by zero.

The nonlinear solver terminated successfully in all cases, but this does not certify a global minimum. The maximum reported raw KKT stationarity residual was about 3.50e-3. Voltage magnitudes remained between 0.9115 and 1.0604 pu, inside the existing estimator's numerical bounds; the probe itself did not impose those bounds.

These results establish numerical feasibility and formulation equivalence on these cases. They do not establish recovery rates or calibrated significance. The fixtures use canonical dispatch and PF, without the full corpus's OPF and operating-limit admission; only direct sensors and buses 4 and 12 were exercised. Candidate flips received ordinary WLS checks, not the full rival scan/certificate. No model correction was applied. One healthy configuration's maximum NLM was 3.1854, illustrating why a blanket maximum-score cutoff of 3 should not be assumed to control false alarms across all coupler constraints.

[Standalone probe](../scripts/probe_exact_coupler_nlm.py) and [complete numerical results](../research/ieee57/coupler_nlm_feasibility_20260912.json).

## Conditions for adopting the recommendation

### Use screening without weakening acceptance

A large injection or flow residual does not uniquely identify an incident branch. Measurement corruption, interacting status errors, missing terminal meters, and weak excitation can change the ranking. NLMs likewise propose coupler hypotheses; they do not authorize a correction by themselves.

The current comparison guard explicitly assumes a hypothesis family fixed independently of the analog measurements. Selecting a top-k family using those same measurements and then budgeting only over the selected rivals changes that assumption. Initially, let NR/NLM scores change the evaluation order while retaining the complete predeclared family for certification. A later pruning scheme needs measured recall, expansion/fallback behavior, and selection-aware statistical justification. Testing a shortlist does not establish uniqueness outside it. See [guard assumptions](../logical_topology/calibration.py#L30).

### Retain nonnested-model and numerical safeguards

An open versus closed coupler exchanges voltage-equality constraints for zero-flow constraints. Equal free-state counts do not make those alternatives nested. Do not interpret every objective difference as chi-square with one degree of freedom. The existing common-relaxation guard uses four possible added real quantities per differing asset CB and two released equalities per differing coupler, with explicit conditional asymptotic assumptions; its finite-sample behavior still needs independent calibration.

Retain the unchanged-model option, both absolute residual checks, all unresolved alternatives, and the complete-scope requirement. Add consistent alternative starts for problematic or competitive nonlinear fits, using current estimates and independent truth-free starts. Different numerical outcomes must remain visible rather than being mislabeled as statistical rejection.

### Treat mixed errors as a separate extension

The current `apply` requires an absolutely plausible candidate. Thus a topology repair that leaves a bad analog measurement or parameter error cannot generally be accepted as an intermediate repair. The raw-section provider also lacks measurement/parameter correction routes.

Joint intervention hypotheses need separate identities and evidence accounting. Report topology recovery separately from whole-episode completion, and preserve all remaining anomaly evidence. WLAV is a reasonable comparison to investigate, but it needs explicit operational-constraint scales, observability checks, and its own calibrated acceptance policy. Its objective cannot simply inherit the WLS chi-square threshold, and dropping more measurements must not become a way to win a comparison.

### Calibrate on independent physical worlds

The copied v2 development audit records 501 correct corrections, zero false CB changes, and preservation of all 314 healthy controls among 1,098 admitted rows. These are historical retrospective development results, not results of the new ranker or a new independent evaluation.

For the next comparison, freeze thresholds and ranking rules on separate healthy worlds that include correctly modeled outages and open couplers. Evaluate direct and indirect sensors, individual and interacting errors, unknown statuses, and mixed analog/parameter overlays. Keep every derivative of one physical parent together across calibration/evaluation splits. Record ranking recall, solver disagreement, unresolved outcomes, false status changes, and final residual consistency.

## Practical next implementation sequence

1. Add an IEEE57 logical diagnostic module that maps normalized residuals to branch candidates and recovers exact constrained coupler NLMs. Keep the current estimator and correction backend.
2. Add numerical diagnostics and consistent multi-start retries; verify constraint scaling/order invariance and healthy noise covariance.
3. Integrate rankings as candidate ordering, retaining the current declared-scope certification and explicit desired-status actions.
4. Measure screening recall and independent false-correction behavior before shortening the search. Add an episode-level expansion and visited-configuration policy when needed.
5. Extend the logical-aware agent protocol for multi-device and mixed-error interventions; add generalized WLAV as a separately assessed comparison.

These are proposed next edits; they were not applied by this review. The source IEEE14 HPC tree remains separate.

## Verification and references

From `C:\dw`, `python -m pytest -q logical_topology scripts/test_audit_logical_topology_artifacts.py` passed **146 tests and 6 subtests in 50.31 seconds**. Module-path checks confirmed imports came from `C:\dw`. Test output (local artifact: `output/ieee57_method_review_20260912/baseline_tests.txt`), test XML (local artifact: `output/ieee57_method_review_20260912/baseline_tests.xml`), and review receipt (local artifact: `output/ieee57_method_review_20260912/review_receipt.json`).

The supplied recommendation is retained as user_method_proposal.txt (local artifact: `output/ieee57_method_review_20260912/user_method_proposal.txt`). The referenced chapter was not attached, so this assessment evaluates the proposal's mathematics and compatibility with the local code; it does not attest to its section/example attributions. The linked MATPOWER implementation was checked directly. Large generated datasets and local audit receipts are intentionally excluded from Git; the portable probe and its compact numerical result are versioned.
