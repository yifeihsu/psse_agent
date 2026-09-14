# IEEE57 logical-switch topology-error adaptation

The implementation follows the supplied review's distinction between two independent connectivity decisions. An asset-status CB includes or excludes a complete line/transformer. A bus-coupler CB joins or separates two explicitly declared synthetic sections of one original bus. The two devices share inspection and correction interfaces, but use different electrical models.

The implementation is in [`logical_topology/`](../logical_topology/README.md). The original full audit is retained at [`output/ieee57_logical_topology_20260911`](../output/ieee57_logical_topology_20260911/report.md); the [version-2 comparison-guard verification](../output/ieee57_logical_topology_20260911_verified_v2/report.md) retains its own output and unchanged physical inputs. These are engineering-development audits, not an untouched final test of a learned policy.

## Scope implemented

| Feature | Implementation |
|---|---|
| Branch-status core | 80 logical asset CBs on the canonical IEEE57 branch rows; inclusion and exclusion in both directions |
| Configuration extension | 13 predefined bus-section partitions and ideal couplers in one versioned layout; split and merging in both directions |
| Healthy controls | Correctly closed networks, correctly modeled branch outages, and correctly open couplers |
| Interacting errors | Nearby/separated pairs, opposite error directions, branch plus coupler, two couplers |
| Mixed errors | Topology plus nearby/distant injection-meter error, or an active branch's R/X error |
| Stress and uncertainty | Unknown status, sparse unobservability, weak indistinguishability, islands, and numerical failures retained explicitly |
| Evidence | Fixed raw meter IDs and covariance; unavailable readings redacted; no measurement regeneration or redispatch during diagnosis |
| Correction | Cumulative current status vector; certificate bound to model and evidence; nonstatus parameters and previous correct statuses preserved |
| IEEE14 bridge | Same logical abstraction; normal-state detailed/logical matrices, admittances and actual PF results match exactly |

The IEEE57 section buses are 1, 4, 6, 9, 11, 12, 13, 15, 24, 38, 41, 49 and 56. Each section receives at least two branch terminals. Generator rows and shunts remain intact on section A; declared loads are split 50/50 before physical measurement generation. Bus 4 uses A={3–4,4–5}, B={4–6,4–18 circuit 1,4–18 circuit 2}. Layout hashes and version IDs identify these synthetic assumptions; no reconstructed physical yard is claimed.

## Executed verification

The completed version-2 run audited all **1,276 scenario/deployment rows**, with **1,098 physically admitted** from **148 of 190 true operating worlds**. Admission retained 170 row views of OPF nonconvergence and eight row views outside the connected-island scope. These correspond to 40 failed-OPF worlds and two islanded worlds; nonconvergence does not prove physical infeasibility.

| Outcome on the identical corpus | Original absolute-fit prototype | Version-2 comparison guard |
|---|---:|---:|
| Applied corrections | 529 | 501 |
| Correct applied corrections | 511 | 501 |
| False CB changes | 18 | 0 |
| Healthy-CB preservation failures | 18 | 0 |
| Exact final status vectors, including healthy controls | 823 | 815 |
| Audit execution failures | 0 | 0 |
| Observation/covariance/nonstatus-parameter preservation failures | 0 | 0 |

The guard declined all **18 previous false corrections** and also declined **10 previously correct corrections** for insufficient separation. All **314 admitted healthy-control rows** retained their correct statuses. The final 815 exact status vectors consist of those 314 healthy controls plus 501 corrected cases; **283 admitted rows retain an unresolved status discrepancy or unknown indication**. Zero false corrections is an observed result on this retrospectively reused corpus, not a claim that every fault was diagnosed or that an unseen nonlinear population has zero error.

In all **28 guard-declined rows**, the returned comparison-compatible set contained the true status vector. Such members need not pass the absolute chi-square/normalized-residual tests. This is empirical coverage on these correlated development rows, not a confidence-coverage guarantee for new cases.

Final single-error recovery was exclusion **166/320**, inclusion **209/244**, split **55/78**, and merging **58/66**. Paired recovery was **13/16** admitted rows. All 36 mixed-error cases were handed off for measurement/parameter investigation, and none of the 12 sparse or 12 unknown-status corpus rows achieved exact final statuses. These outcomes remain in the results rather than being filtered out.

The initial evidence tests flagged **663 chi-square alarms**, **692 normalized-residual alarms**, including **46 normalized-residual-only alarms**. Revalidation returned **36,019 verified historical numerical fits** and performed **624 new WLS solves** for additional candidate checks. It regenerated no physical worlds or observations. Source snapshots and both original/copied corpus bytes remained unchanged throughout the run.

The final regression suite passed **146 tests plus six subtests** (99 package tests and 47 independent-checker tests). The [test receipt](../output/ieee57_logical_topology_20260911_verified_v2/implementation_checks_final.json), [prototype comparison](../output/ieee57_logical_topology_20260911_verified_v2/comparison_to_prototype.json), and [full run receipt](../output/ieee57_logical_topology_20260911_verified_v2/run_receipt.json) retain the executed counts and hashes.

The [full independent artifact audit](../output/ieee57_logical_topology_20260911_verified_v2/independent_artifact_audit.json) **passed with zero failures**. It checked all 1,098 detailed audits and 152,730 candidate records, independently recomputed the final outcome counts, verified all 1,098 guards as version 2, and checked numerical reuse against original and current inputs. Eight physical PF replay groups and 15 stored GSE-state replays passed: maximum physical measurement discrepancy was 5.82e−11 pu, and stored-state prediction discrepancy was zero. Executed source snapshots, current core sources, and original/copied corpus hashes matched.

## Electrical and measurement formulation

For a branch, status multiplies the complete terminal admittance, including charging and transformer tap/shift effects. Status zero means the whole asset is disconnected at both ends. The implementation deliberately does not model one-end-open energized lines, whose remaining charging would require terminal-specific connectivity. [MATPOWER implementation](https://matpower.org/docs/ref/matpower7.1/lib/makeYbus.html).

Closed couplers use exact node contraction. Open couplers retain independent section voltages and zero coupler flow. No small-impedance surrogate or merge between unrelated original buses is introduced. Each model retains immutable branch/equipment identities and explicit physical-node-to-numerical-bus mappings.

Branch-only measurements contain 491 channels: Vm, Pinj and Qinj at 57 buses, and P/Q at both ends of 80 branches. The section layout has 70 physical measurement nodes and 530 analog records. Available terminal-flow readings remain present even when the estimator reports a branch disconnected. Direct and predeclared even/odd indirect deployments share available-channel noise draws; each indirect profile omits all four flow readings on half the branches. Error targets are selected from that already-masked set. Unavailable operator values are `None`, not simulated hidden direct evidence.

Section injection readings are calculated from assigned physical generators and loads after the true PF. They are not formed by copying or dividing a noisy bus-total observation. Legitimate aggregation retains raw records and propagates covariance as `R_agg = A R_raw A.T`; voltage averaging across sections is rejected.

For each closed coupler, the estimator fits two P/Q flow nuisance variables while using one shared voltage. Opening that coupler adds two voltage degrees of freedom and removes two flow variables. The connected full section formulation therefore has 139 free states, regardless of which couplers are open. With all 530 records available, its residual degrees of freedom are 391. The branch-only formulation has 113 states and 378 degrees of freedom. Actual weighted-Jacobian rank, available rows and covariance determine each test; dimensions are not inferred merely from the original 57-bus case.

WLS starts at unit voltage and zero angles/flows. The initial detection tests remain chi-square alpha=0.05 and maximum normalized residual 4.0. Off-branch zero-flow rows have zero model sensitivity and are retained in both residual tests. A proven nonzero flow on a fully excluded branch can reject a hypothesis without executing a redundant fit; the audit labels that as an analytical proof and does not invent a solver result.

## Identifiability and the comparison refinement

The first full development sweep exposed a false correction on a correctly modeled outage with indirect evidence. The current model had J=254.738 against a chi-square threshold of 253.444. A wrong extra outage reduced J to 250.857, while normalized residuals remained below four. Exactly one candidate passed the absolute thresholds, but a gain of only 3.88 did not establish a reliable topology distinction.

This motivates a separate model-comparison gate in addition to absolute goodness of fit. The review's four directions are not interchangeable continuous scalar-parameter tests. For a chosen candidate and rival, a common relaxed measurement model can contain both hypotheses:

- A differing branch status can be embedded using four free real terminal P/Q perturbations.
- A differing coupler status can be embedded by releasing its two voltage-equality or zero-flow constraints.

If `d = 4 * differing_branch_CBs + 2 * differing_couplers`, then

`J_rival - J_chosen <= J_rival - J_common`.

Under the regular nested-model assumptions, the right-hand side has an asymptotic chi-square law with rank gain no greater than d. Using the larger d and a Bonferroni allocation across the declared candidate family provides a conservative large-sample comparison. Strong zero-flow contradictions also admit a direct Gaussian tail test with correction for witness-row selection. Weak analytical witnesses need a full rival fit rather than being treated as automatically significant.

The two rejection routes share the family budget. With N declared models and family alpha=0.05, each route receives alpha/2 and each rival comparison receives alpha/[2(N−1)]. The fitted-model critical gain is therefore `chi2.isf(alpha/[2(N−1)], d)`. Both reported adjusted p bounds include this factor of two, and Gaussian witnesses also include the number of eligible zero-flow readings. Falling back to the fitted-model test after a weak Gaussian witness cannot spend a second full family budget. Version-2 certificates bind this allocation; earlier certificates cannot authorize a correction.

This is conditional on known covariance, correctly specified analog/parameter models, identifiable interior solutions, a predeclared candidate family, and adequately minimized objectives. It is not an exact finite-sample guarantee for nonlinear power-system estimation. Gross-meter and parameter-error mixtures can violate the pure-topology model assumptions; their false corrections remain measured outcomes, not excluded samples. The source of the large-sample likelihood-ratio approximation is [Wilks, 1938](https://doi.org/10.1214/aoms/1177732360); the common-model bound above is the implementation's formulation-specific derivation.

The refinement is evaluated retrospectively on identical physical cases and observations. Previously computed WLS results may be reused only after exact input and estimator-source verification. No physical truth is regenerated and no root is dropped because the revised diagnostic declines to correct it. The earlier result remains preserved as development evidence. An intermediate full revalidation was stopped when the independent statistical review identified the need to share the two testing routes' budget; its partial artifacts are marked incomplete.

## Three separate audits and preservation

Every planned candidate receives physical feasibility, state-observability and status-identifiability outcomes. Physical admission uses the true topology, AC OPF, a consistent AC PF, generator/voltage/declared-rating checks and equation residuals. It does not use wrong-model WLS convergence or teacher success. Islands receive an explicit separate category; no load is silently shed and no extra slack is introduced.

The runtime receives only the reported topology and fixed observed sensor records. Truth is loaded only after its decisions for offline scoring. Reports include exact status recovery, false changes, healthy-CB preservation, state-estimation error, and byte/semantic preservation of observations, covariance and nonstatus case fields. Topology recovery in a mixed case does not imply that the remaining meter or parameter error was repaired.

All noise, measurement-profile and reported-error derivatives of one true operating world share `parent_physical_root`. Structural holdout reserves entire shared parents. This correctly prevents leakage but can reserve all closed-world exclusion/split cases for test, leaving those training families empty. The generated split views therefore require coverage review and additional independent operating worlds before training; they are not a training-readiness certificate.

The executed coverage has explicit limits. All 36 mixed-error rows use direct telemetry and true-closed/model-open topology discrepancies. All 12 corpus rows with unknown status have actually closed devices; an additional runtime regression covers an actually open branch. Ambiguous and weakly distinguishable outcomes are retained, but this corpus does not contain a dedicated sweep of low branch flow or small section-voltage differences. These dimensions remain separate from the complete single-branch and single-coupler direction coverage.

The original manifest's parameter-overlay field `physically_identifiable_label` checks whether the affected branch is active in the true system. It prevents an overlay on a disconnected asset; it does not establish parameter identifiability. The [independent coverage receipt](../output/ieee57_logical_topology_20260911_verified_v2/coverage_review.json) records these limits, the full pair-search scopes, and missing structural-training families against matching original/copied manifest hashes.

## Existing pipeline bridge

The new opt-in provider uses the existing state-store and action/modification conventions. Actual tests cover WLS → certified topology context → model correction → candidate WLS for a 530-channel coupler case, including a 58-to-57 electrical-bus transition while preserving all raw records, covariance, an existing branch outage and prior R/X edits.

The final provider review also repaired preservation of current branch angle limits. All nonstatus branch fields now survive topology correction, and changed parameters invalidate older certificates. Current bus, generator, base-MVA and cost edits must match canonical metadata; inconsistent edits are rejected explicitly because their section allocation cannot be inferred safely.

The old release factory/private truth audit still assumes case-sized bus vectors and branch-status targets. Logical section models are explicitly guarded from that path. Coupler release/DAgger integration and learned-policy evaluation are not claimed by these provider-hook tests.

A separate four-case direct branch-only bridge ran the unchanged existing observable expert. It achieved 3/4 audited task successes. The exclusion case changed healthy measurement index 411 and left the wrong branch status in place. Strict offline auditing caught the damage even though the legacy `false_commit_count` remained zero. The complete trace and measurements are retained in [the legacy bridge report](../output/ieee57_logical_topology_20260911/legacy_branch_bridge/BRIDGE_NOTES.md). Its outcomes did not determine source-corpus admission.

## Reproduction

```powershell
python -m pytest logical_topology scripts/test_audit_logical_topology_artifacts.py -q -p no:faulthandler
python scripts/validate_logical_topology.py --output-dir output/logical_topology_new --system case57 --preset full --load-scales 0.8 1.0 --workers 4
```

To repeat the guarded comparison on the fixed original physical corpus:

```powershell
python scripts/revalidate_logical_topology.py --source-run output/ieee57_logical_topology_20260911 --output-dir output/logical_topology_revalidation_new --workers 4 --max-pairs 5000
python scripts/audit_logical_topology_artifacts.py --output-dir output/logical_topology_revalidation_new
```

Use new output directories. Generation and revalidation archive their executed sources, including the comparison guard, and record numerical dependency versions. The historical original run did not attest those versions; reused fits retain that explicit limitation rather than inventing a matching historical environment.

The [`logical_topology` README](../logical_topology/README.md) documents individual inventory, measurement, estimation, runtime and provider APIs. Source snapshots, true-world admission records, fixed input files, complete candidate scans and compact row audits accompany each executed corpus.
