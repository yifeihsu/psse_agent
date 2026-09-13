# Logical topology-error testbed

This package implements the same logical-switch abstraction for the pinned IEEE14 and IEEE57 cases. It represents branch inclusion/exclusion using one **asset-status CB per canonical branch row**, and split/merge errors using separate **ideal couplers between two sections of the same synthetic substation**.

| Layout | Asset-status CBs | Bus couplers | Physical measurement nodes | Raw analog channels |
|---|---:|---:|---:|---:|
| IEEE14 branch only | 20 | 0 | 14 | 122 |
| IEEE14 sections | 20 | 5 | 19 | 137 |
| IEEE57 branch only | 80 | 0 | 57 | 491 |
| IEEE57 sections | 80 | 13 | 70 | 530 |

An asset CB at status zero removes the entire branch, including charging at both ends. It is not a physical terminal breaker and does not cover an open-ended, still-energized line. Parallel circuits, tap=1 transformers, and original branch row IDs remain distinct.

The section layout is synthetic and versioned. Every selected bus has at least two incident branch terminals on each section. Generators and shunts remain intact on a declared section; aggregated loads are allocated explicitly before measurement generation, preserving P and Q. Bus 4 uses the review's A={3–4,4–5}, B={4–6,4–18 circuit 1,4–18 circuit 2} assignment. Closing a coupler contracts the sections exactly; opening it retains two electrical buses. No tiny-impedance lines are inserted.

## Generate and test

```powershell
python scripts/validate_logical_topology.py --output-dir output/logical_topology_new --system case57 --preset full --load-scales 0.8 1.0 --workers 4
```

`--preset smoke` exercises a smaller selection and deliberately limits two-error searches to 32 pairs; incomplete pair searches cannot authorize corrections. The full preset evaluates the complete two-CB neighborhood (up to 5,000 pairs). Existing output directories are preserved.

The source generator:

1. Freezes logical equipment layouts, physical meter identities, and availability masks.
2. Solves each **true** topology with AC OPF followed by a consistent PF and physical/bounds checks.
3. Generates one observation realization per true operating world; direct/indirect deployments share the same available-channel noise draws.
4. Changes only reported statuses and optional declared measurement/parameter overlays.
5. Retains every candidate and physical-admission reason before running WLS or the rule-based diagnostic.

Families include inclusion, exclusion, split, merging, correctly open/closed controls, unknown status, nearby/distant bad-data overlays, topology plus active-branch parameter errors, nearby/separated paired errors, branch-plus-coupler errors, and a sparse unobservable profile.

## Core API

```python
from psse_env.systems import resolve_system
from logical_topology import build_inventory, process_topology

case = resolve_system("case57").load_case()
inventory = build_inventory("case57")  # split_buses=[] selects branch only
statuses = dict(inventory["normal_statuses"])
statuses["case57:bus:4:coupler"] = 0
compiled = process_topology(case, inventory, statuses)
assert len(compiled["case"]["bus"]) == 58
```

Keep `case` on its original equipment/base-bus basis. The compiled case has temporary numerical endpoints and is accompanied by explicit `node_to_bus` and `node_to_row0` mappings. Current R/X and other case edits are preserved during cumulative status changes.

```python
from logical_topology import LogicalTopologyRuntime

runtime = LogicalTopologyRuntime(
    inventory=inventory,
    current_case=current_base_equipment_case,
    current_statuses=reported_statuses,
    measurement_inventory=fixed_sensors,
    observations=fixed_observations,
)
details = runtime.inspect_cb("case57:bus:4:coupler")
trial = runtime.test_cb("case57:bus:4:coupler", 1)
scan = runtime.scan_candidates()
if scan["unique_candidate_id"] is not None:
    corrected = runtime.apply(scan["unique_candidate_id"])
```

The runtime receives no physical truth, error labels, clean measurements, or solved true-state initializer. `test_cb()` is nonmutating. Application requires a certificate bound to the current model, observations, covariance and complete declared hypothesis scope, plus the comparison guard described below. A plausible current configuration is retained even if it differs from the all-closed reference.

## Measurements and statistical tests

Physical record IDs survive topology processing. Available flow rows remain present even for a modeled or actually disconnected branch. Unavailable sensors retain their IDs and covariance metadata, with `None` values at the operator boundary. Even/odd masks suppress half of the branch-flow instruments before the erroneous asset is selected; the missing pattern cannot identify one particular faulty branch.

Section injection observations are actual assigned generator-minus-load quantities, not independent copies of a bus-total meter. Legitimate aggregation uses `A z` and `A R A.T` and retains the original records. Unavailable rows cannot be aggregated; voltages from separate sections are not averaged.

For each closed coupler, the estimator uses one shared bus voltage and two unmeasured P/Q flow variables. Opening it adds two voltage degrees of freedom and removes the two flow variables. Thus the full IEEE57 section layout has 139 free states under connected known-status candidates, and 391 residual degrees of freedom with all 530 sensors. The Jacobian rank is checked rather than assumed. No derived coupler flow is counted as an independent sensor.

WLS starts flat. Candidate plausibility requires convergence, full state observability, a rank-aware chi-square test at alpha=0.05, and maximum normalized residual below 4.0. All retained candidates use identical physical evidence. Weak, ambiguous, numerical-failure and unobservable alternatives are explicit outcomes; minimum residual alone does not authorize a correction.

An available nonzero flow on a candidate's fully disconnected branch supplies a rigorous rejection shortcut: `h_i=H_i=0`, hence its normalized residual is `abs(z_i)/sqrt(R_ii)` for any state. These records are reported as analytical rejections, not fabricated WLS solves. The initial model still receives a full WLS attempt.

Before applying an otherwise unique candidate, `calibration.calibrate_scan()` also checks separation from every rival. Fitted models use a conservative common-relaxation likelihood-ratio envelope with four extra real terminal-flow quantities per differing branch CB and two released constraints per differing coupler. The alpha=0.05 family budget is split equally between that comparison and the Gaussian zero-flow route, then across the whole declared rival family. Gaussian witnesses also account for witness-row selection. Weak witnesses trigger full rival fits; insufficient separation produces an explicit compatible candidate set and no correction. Basic, stale or earlier-version certificates cannot bypass this guard.

The comparison is conditional and asymptotic for regular, correctly specified models with known covariance and adequate numerical fits. It is not an exact nonlinear false-correction guarantee, especially for unmodeled gross-measurement or parameter errors. These conditions and unverified assumptions are included in every guard result.

## Provider and audit boundaries

`logical_topology.provider.LogicalTopologyProviders` exposes opt-in WLS, topology-context, and topology-correction hooks using the existing state-store/action/modification conventions. A 530-channel coupler-correction chain is tested through `PowerSystemStateStore`, including preservation of prior R/X edits and a correctly modeled branch outage.

Current branch parameters, including angle limits, are preserved by canonical row identity; changing them invalidates a previous correction certificate. Bus, generator, base-MVA and cost changes must agree with the explicit canonical base-case metadata and section allocation. The provider rejects mismatches rather than silently restoring stale values.

Use `state_payload()` to prepare inputs and `provider_hooks()` to obtain the supported hooks. The legacy measurement/parameter correction routes and `env_kwargs()` factory are intentionally rejected for this raw-section schema: the old protocol/private truth audit assumes branch-status targets and a case-sized measurement vector. These hooks do not claim full DAgger/release-factory or learned-policy integration for couplers. The standalone runtime and its offline status/evidence audits are the executed logical-topology path.

The corpus distinguishes physical feasibility, true-model state observability, and status identifiability within declared candidate hypotheses. Physical rejection is not inferred from a failed wrong-model WLS. Outcome reports include final status accuracy, false corrections, healthy-device preservation, unchanged measurements/covariance/parameters, and state-estimation errors. Mixed-error reports do not equate topology recovery with repair of remaining meter or parameter errors.

`parent_physical_root` groups every deployment, noise, model-error and overlay derivative of the same true operating world. `physical_root_fingerprint` is the declared noise-invariant topology-error identity; `scenario_id` also includes the noise seed/profile. Structural holdout reserves whole shared parents, which can leave some training families sparse or absent. Split views require coverage review and further independently generated operating conditions before training.

See the [comparison-guard verification report](../output/ieee57_logical_topology_20260911_verified_v2/report.md), [original development sweep](../output/ieee57_logical_topology_20260911/report.md), and [detailed implementation notes](../docs/ieee57_logical_topology_20260911.md). The refinement keeps the original corpus unchanged and reuses numerical fits only after exact model/evidence and numerical-source checks. It never reuses an old correction decision or certificate. The IEEE14 normal-state bridge compares the logical and detailed models; it does not assert equivalence of their different terminal-switch fault semantics.

The completed v2 audit covers 1,276 rows, including 1,098 physically admitted rows: 501 correct corrections, zero false CB changes, and all 314 healthy-control rows preserved. The guard declines all 18 earlier false corrections and ten earlier correct corrections; 283 admitted rows retain unresolved statuses. The independent artifact audit passed, as did 146 tests plus six subtests. These are retrospective development results on the same physical corpus, not an untouched learned-policy test.

To reproduce the comparison refinement against an already completed source run, use a new output directory:

```powershell
python scripts/revalidate_logical_topology.py --source-run output/ieee57_logical_topology_20260911 --output-dir output/logical_topology_revalidation_new --workers 4 --max-pairs 5000
python scripts/audit_logical_topology_artifacts.py --output-dir output/logical_topology_revalidation_new
```

The run receipt distinguishes reused historical numerical fits from new estimator calls and records any missing historical dependency-version attestation. Each diagnostic decision and correction certificate is rebuilt. Physical OPF/PF admission results are retained from the original generation; independent artifact replay checks are recorded separately.
