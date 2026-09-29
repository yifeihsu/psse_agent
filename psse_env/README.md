# Constrained recovery-aware DAgger

`psse_env` is the transactional DAgger layer for PSSE diagnosis. Its central
contract is:

```text
policy-only observation
  -> safe action normalization
  -> deterministic process-validity gate
  -> transactional candidate branch
  -> mandatory verification
  -> candidate assessment
  -> commit or rollback
```

`TransactionalPSSEEnv.get_policy_observation()` returns deployment-visible
state only. `get_oracle_state()` adds synthetic truth and action hints for the
expert. Chat conversion fails closed if a forbidden oracle key reaches the
user prompt.

State IDs are episode-namespaced. Invalid calls are standardized no-op
transitions and remain in rollout history. Corrections create candidates;
only verified accepted candidates may commit, and only verified rejected or
inconclusive candidates may roll back.

Correction arguments are canonicalized before the process gate. The legacy
`arguments.modification` wrapper is flattened, conflicting nested/outer
targets fail closed, and list-form measurement updates become an index-to-value
mapping. A measurement macro must use explicit indexed updates; whole-vector
replacement is outside the bounded action contract. Context and evidence
providers receive a copied physical state payload (`case`, `measurements`, and
provenance) plus a nested deployment-safe `policy_observation`.

`psse_env/providers/matpower.py` supplies deployment WLS/context/correction
adapters (`provider_kind="deployment"`) backed by the same pure Python
estimation stack as the production MCP server: Lagrangian WLS with residual
and branch-multiplier evidence, chi-square global tests, grouped
`suspect_group` measurement correction, multi-scan parameter correction (scans
come from `metadata.parameter_scans`), and branch-status topology correction
via content-addressed derived case files. `MatpowerDeploymentProviders().
env_kwargs()` wires the bundle plus a `ProcessValidityOracle` accepting
executor-hydrated (target-only) corrections and a deployment
`CandidateQualityOracle` with a MATPOWER case differ for path-valued cases.
Candidate verification derives observable `target_fixed` and
`remaining_suspect_count` evidence from the candidate solve alone. The latter
counts thresholded residual/multiplier suspects, not physical faults; final
versus partial deployment acceptance uses target improvement plus the global
anomaly test. Synthetic truth remains separate as
`remaining_true_fault_count`.

The six specialized diagnostics — `get_harmonic_context`,
`get_three_phase_context`, `run_hse_from_path`, `run_three_phase_nlm_from_path`, and both HIF
estimators — are first-class macro actions sharing their canonical deployment
names. The process gate treats them as read-only evidence actions (legal on
the active state, or on an INCONCLUSIVE candidate), and the environment
dispatches them through `evidence_providers` with the full action so bounded
estimator arguments (candidate branch, phase, grid options) reach the tool.
The deployment bundle wraps the real HSE, three-phase NLM, and HIF estimation
stacks; runtime side data comes from state metadata
(`harmonic_measurements`/`harmonic_orders`, `three_phase_voltages`,
`nlm_diagnostic` or OpenDSS model dirs, `hif_runtime`, `hif_scan_window`). A
harmonic-context request without spectral measurements succeeds with
`harmonic_context_status=unavailable`; it does not classify the network as
clean or explain its anomaly. Estimators fail closed as collectable no-ops
when their required data is absent. The protocol bridge maps
`state_id` to `case_path` (or `scan_window_path` for the multi-scan
estimator) so exported targets and generated calls stay canonical.

An unflagged root exposes only positive-sequence measurements and model
information initially. `PolicyObservation.available_evidence` does not
advertise private spectral data or its availability at reset. After a WLS
anomaly, `get_harmonic_context` explicitly requests additional measured
spectra and reports which channels were acquired. Spectral measurements are
not inferred or fabricated from the positive-sequence snapshot. The same
request is proposed for the same observable WLS anomaly whether or not the
provider has spectra, and independent of the hidden fault family.

`DiagnosticsExpert` routes on acquired telemetry and observable signatures.
A harmonic-context result mints a harmonic signature only when the returned
measurements pass its distortion/noise screen; the expert then calls
`run_hse_from_path`. Pure three-phase-unbalance
signals stop at a VUF/null-gated non-HIF classification; HIF-specific signals escalate
`run_three_phase_nlm_from_path` -> the multi-scan estimator when a persistent
scan window exists, else the single-scan estimator, carrying the NLM top
branch as `candidate_branch_row0`. Privileged fault families and hints are
ignored by this expert, so changing hidden truth while holding the policy
observation fixed cannot change the production target.

`three_phase_branch_currents` (per-phase terminal current phasors on every
branch, with `branch_current_sigma_pu`) is a further observable channel. When
it is present, `run_three_phase_nlm_from_path` localizes an HIF line and phase
from the two-terminal differential current measured on the snapshot (averaged
coherently across a scan window), reports a closed-form
`terminal_current_estimate` of position and resistance, and keeps any stored
NLM diagnostic only as secondary evidence; the expert forwards the observed
`suspected_phase` as `candidate_phase`, and both HIF estimators seed their
OpenDSS search from the closed form and add a current residual block. Both
estimators may accept an HIF explanation on the strength of the differential
itself (`acceptance_basis=terminal_current_differential`) when it clears the
six-sigma floor, the closed-form fault impedance is positive and resistive,
the two terminals agree on the fault-point voltage, and the model search
agrees on the phase; the residual-reduction gate is diluted by sensor noise on
hundreds of unaffected entries and remains the other accepted basis. For a
pure unbalance signature the same tool localizes the *source bus* by the
per-phase shunt-power spread computed from KCL (negative-sequence voltage
alone peaks at weak buses, not at the source), records it as `bus_1based` in
the explanation for the release audit, and accepts the explanation when that
spread is significant against the current-sensor noise and no line carries a
differential current above the sensor floor (an explicit non-HIF null); the
VUF gate is reported as `voltage_gate_passed` but no longer decides, because
its 2% threshold was calibrated on a corpus whose Bus 3 was always unbalanced.
The channel satisfies the production-row telemetry gate for these tools on
its own. Rows without the channel behave exactly as before.
Diagnostic summaries (`wls_summary`, `hse_summary`, `nlm_summary`,
`hif_summary`, `diagnostic_acceptance`, ...) are model-visible history metrics
in SFT export. The production target audit independently requires matching
observable signature provenance and telemetry, and binds HIF-estimator branch
targets to the latest successful NLM output.

Diagnostic findings resolve anomalies without a physical correction through
explained-anomaly records. A provider declares an `anomaly_explanation`
only after an explicit null/goodness gate accepts the finding: HSE requires
THD above its configured threshold, the unbalance path requires VUF above its
configured threshold, and HIF estimation requires material improvement over
the no-HIF model with acceptable residual fit. A best candidate or successful
optimizer alone is not terminal evidence. The environment binds an accepted finding to
the unresolved signatures matching that family's markers
(`ANOMALY_FAMILY_MARKERS` in `actions.py`, shared with expert routing) and
records it in the model-visible `explained_anomalies` field. Once every
unresolved signature is covered by an explanation, the terminal condition is
met: the process gate legalizes `finalize_diagnosis`, the termination expert
proposes it (`anomalies_explained_by_diagnostics`), and the production
finalize audit accepts it when each contributing record carries an
observable evidence source. Diagnosed-but-uncorrected episodes therefore
terminate cleanly instead of stalling on a persistent chi-square anomaly.

Deployment `run_wls` also refreshes the model-visible
`unresolved_signatures` from the solve itself: sensor-sourced signatures are
preserved, and when the chi-square test fires it mints residual-outlier and
branch-multiplier signatures from the top residual/λ evidence. Classical λ-vs-r
dominance discrimination tags the dominant family: `max|r| > 1.2·max|λ|` marks
`wls_residual_outlier_dominant`, `max|λ| > 1.2·max|r|` marks
`wls_branch_multiplier_dominant`, and inside the symmetric dead band neither
carries the token, so routing falls back to static source priority
(parameter → topology → measurement). Family experts boost their context
confidence on a dominant signature (`dominance_confidence`), and the
measurement expert stands down while branch evidence is dominant — until both
branch families have had a hypothesis rejected by verification — because a measurement
correction can zero the residuals of a wrong model and mask a branch fault.
Two more physical guards close that masking channel: while a
harmonic/unbalance/HIF sensor signature stands, explained or not, `run_wls`
mints no `wls_*` signatures at all, the three context providers offer no
supported corrections (findings stay visible as evidence, with
`fundamental_route_blocked_by_waveform_anomaly` naming the signatures), the
process gate refuses every correction as `correction_route_not_actionable`,
the recovery expert defers to the diagnostic ladder instead of its generic
WLS fallback, and the classical family experts stand down in the combined
stage. An accepted explanation closes the episode's obligation; it does not
remove the event from the network, so the fundamental-frequency solve stays
unreliable for as long as the signature is present, and explanation-only
families terminate by diagnosis or operator handoff, never by repair.
Separately, `get_topology_context` filters supported status flips that would
island the network (an EMS would never offer that switching action).

Waveform roots come in two signature modes. `flagged` seeds the sensor
signature at reset, as if a power-quality monitor or a zero-sequence relay
had raised it. `discovered` withholds it. Harmonic roots default to
`discovered`: the operator starts from positive-sequence measurements and
the model, runs `run_wls`, then requests additional spectral measurements
through `get_harmonic_context` if WLS reports an anomaly. On the current
synthetic harmonic roots the expert sequence is `run_wls` ->
`get_harmonic_context` -> `run_hse_from_path` -> `finalize_diagnosis`.
The WLS anomaly justifies further investigation; it does not identify
harmonics. HSE requires same-state acquired spectral evidence and a measured
harmonic signature, so a learner cannot skip the initial WLS/acquisition
steps. A clean WLS control finalizes without requesting spectra. If spectra
are unavailable, the anomaly remains unresolved and ordinary investigation
continues without a harmonic diagnosis. An accepted HSE explanation covers
the discovered harmonic signature and the preceding WLS anomaly.

For unbalance discovery, three-phase channel availability also stays hidden
until an explicit `get_three_phase_context` request succeeds. The default
expert sequence on the current unbalance roots is `run_wls` ->
`get_three_phase_context` -> `run_three_phase_nlm_from_path` ->
`finalize_diagnosis`. Both measurement requests follow an observable WLS
anomaly regardless of the hidden family or provider-side data availability;
which one goes first is decided by the residual breadth the solve reports
(`anomaly_breadth`, the share of normalized residuals above the outlier
threshold, kept in the durable WLS ledger). Spectral distortion makes the
operator vector inconsistent with the balanced model almost everywhere
(75 to 84 of 122 channels on the corrected corpora), so a broad anomaly (at
least one half) requests spectra first; a load unbalance, a bad meter, or a
branch fault stays narrow (at most 49) and requests phase measurements
first. Either request falls back to the other when it returns nothing, so
a classical root sees both requests before ordinary investigation. The
three-phase request reports measured coverage without diagnosing or
localizing a fault. If no phase measurements are available, its successful
`unavailable` response leaves the WLS anomaly unresolved and permits ordinary
investigation.

For non-HIF paths, runtime execution and training-label gates both require
successful current-state WLS and fresh acquired three-phase measurements
before NLM. A failed early call cannot expose phase channels or count as a
completed acquisition. State changes and failed refreshes invalidate the
acquired evidence; the request and WLS records survive bounded model history.
Explicit legacy HIF sensor flags retain their existing diagnostic route.

The orchestrator screens acquired three-phase telemetry with
`run_three_phase_nlm_from_path` before correction. Screening
classifies the three-phase state as `balanced_three_phase`
(no explanation; the classical routes stand), an unbalance source
(explanation recorded, the diagnostic mints its own
`three_phase_unbalance localized_by_diagnostic` signature and covers the
`wls_*` signatures minted before the event was known), or `hif_suspected`
(the provider mints `hif_suspected_line_differential` and the ordinary HIF
ladder takes over). The generator defaults harmonic and unbalance to
`discovered`, and HIF to `flagged`; the physical root
fingerprint ignores signatures, so the two modes share roots. A candidate
whose verification solve itself fails is recorded as verified-REJECT — the
solver failure is observable rejection evidence — so the episode retains a
legal rollback path instead of deadlocking on an unverifiable candidate.
After a candidate passes the global WLS chi-square test, the deployment
provider emits a separate `steady_state_physical_evidence` record scoped to
the observed snapshot: connectivity of the in-service MATPOWER topology,
measured bus `Vm` against `VMIN`/`VMAX`, and measured terminal MVA against
positive `RATE_A` limits on active branches. This is not a power-flow
convergence claim. Complete violations set `physical_constraints_ok=false`;
missing or malformed inputs leave it null/inconclusive, so acceptance remains
fail-closed. Topology fixtures clamp PYPOWER generator voltage setpoints to
their declared bus bounds before synthesis.

Candidate acceptance and episode finality are separate decisions. In
production, a successful correction followed by a quiescent WLS statistic is
evidence that the transaction may be committed, but it is not an independent
certificate that every physical error has been removed. The controller keeps
an observable post-correction confirmation obligation, requests same-state
context, and hands off to an operator if no independently supported autonomous
route remains. It never exposes or consumes the private truth audit to choose
that action. Clean states and explanation-only diagnostic states with no
accepted correction retain their existing finalization routes.

The post-correction confirmation *boundary* has one canonical policy-visible
predicate, owned by `oracle/process_validity.py` and reused by the recovery
probe generator. The verified terminal measurement-closure attestation also
has one canonical observable parser,
`oracle/measurement_recovery_evidence.py:verified_terminal_measurement_closure_action`.
The environment, expert, and private post-target audit reuse that parser. The
private audit adds only truth-ledger membership and physical-safety checks after
the observable attestation has been accepted; it cannot reinterpret or widen
the observable closure contract.

This changed finality behavior is explicitly versioned as supervision policy
`bc0_observable_sequential_handoff_v2` and expert identity
`bc0-observable-handoff-expert-v2`; DAgger-1 binds the corresponding
`dagger1_observable_recovery_handoff_v2` collector contract. The pinned family
numeric floors and ceilings are not relaxed. Release policy v3 instead names
the quantity that can be supported without a privileged production label:
audited completion is either strict physical resolution or a state-bound
post-correction controller handoff whose separate private completion audit
passes. The actual outcome remains `operator_escalation`; partial, HIF, and
generic handoffs do not enter the audited-completion numerator.

## Round-0 expert aggregate and the BC0 evaluation suite

`providers/scenario_generator.py` builds the round-0 offline aggregate from
real physics. `Round0ScenarioGenerator` adapts the merged measurement corpus
(single and multi gross outliers, corrupted-parameter cases with multi-scan
data, harmonic and HIF rows with their runtime side channels), synthesizes
topology scenarios through pypower power flows on status-flipped IEEE-14
cases, and composes measurement overlays with other families. The generator
supports twelve scenario capabilities, but the BC0 default aggregate and
frozen evaluation policy select ten: no-error, measurement,
multi-measurement, parameter, topology, harmonic, HIF,
measurement+parameter, measurement+topology, and measurement+HIF.
Three-phase unbalance and telemetry-no-disturbance remain supported generator
capabilities outside the BC0 family policy; a future freeze must add explicit
quotas and thresholds before claiming either one.

Outside the frozen release path, `scripts/run_dagger_research.py
--plan-preset diagnostic` (or `combined`) collects a research round on the
explanation-only families: HIF, measurement+HIF, three-phase unbalance, and
the balanced telemetry control. It draws from the per-phase branch-current
corpora, emits unbalance sensor signatures only when the row's telemetry
actually shows them, and runs the OpenDSS estimators under the validated
research budget through a research-only environment factory; the release
factory module is not modified. See
`docs/branch_current_telemetry_20260903.md`. Every selected scenario
passes its physical validation gate, and scenario IDs are opaque hashes;
family, cardinality, network case, and source tier remain audit/split metadata
rather than policy-visible hints.

Harmonic roots can be included through explicit `--train-plan` and
`--development-plan` quotas. `--harmonic-signature-mode discovered` is the
default; `flagged` is retained only for reproducing legacy monitor-alarm
roots. The mode is recorded in the research configuration even for a
harmonic-only plan, while withheld signatures stay in the private release
audit. Changing the mode cannot silently resume an existing run with a
different recorded configuration. Existing collected traces and trained
checkpoints are unchanged by this generator/policy revision; new collection
is needed to train or evaluate the WLS-first behavior.

`examples/generate_round0_aggregate.py` is expert-only collection at
`β=1.0`. It produces a candidate BC0 behavioral-cloning corpus, not a DAgger
iteration. Before the environment, policy, or online expert receives a root,
the collector deep-copies it and removes `true_*`, `clean_*`, `hidden_truth`,
and oracle action hints. The original scenario is retained outside the online
trajectory and is supplied only after termination to the strict offline audit
in `dagger/release_audit.py`; audit truth and audit results are never merged
into model observations or SFT targets.

BC0 uses the machine-readable supervision policy
`bc0_observable_sequential_handoff_v2`. At each expert-controlled state, the policy
supervises only the current rank-one action in the deterministic observable
protocol. Other process-valid proposals remain in the raw row as
`deferred_expert_actions`; they become eligible only after the preceding
action is exhausted or observably rejected. This is sequencing, not a claimed
Q-cost estimate, and the collector refuses to use the contract outside
production mode, iteration 0, and `beta=1.0`.

BC0 therefore does not require `rollback_state` or
`rollback_state × rejected_candidate_recovery` support. A rejected learner
candidate cannot occur naturally when every action is selected and executed by
the expert. Those two floors begin with DAgger iteration 1: at least ten
independent `physical_root_fingerprint` values must contain a
production-label-eligible `rollback_state` target in
`rejected_candidate_recovery`. The candidate verdict must come from the
deployment-mode oracle after a learner-controlled or explicitly observable
recovery-probe action. Truth-derived `synthetic_counterfactual` rows and
unranked multi-action auxiliary rows never satisfy that gate. The
machine-readable next-phase contract is
`DAGGER_ITERATION_1_RECOVERY_GATE_POLICY` in
`examples/generate_round0_aggregate.py`; BC0 provenance records it without
applying it to the round-0 training view.

Regeneration uses the tracked ten-family row-budget plan rather
than `--scale 2`, which would incorrectly request 34 HIF roots from the
17-root training inventory:

```bash
python -m psse_env.examples.generate_round0_aggregate \
  --plan data/round0_plan_20260719.json \
  --output-dir data/round0_aggregate_release
```

The generator publishes `SHA256SUMS` last, covering every JSON/JSONL artifact
in that directory.

That plan has 263 roots.  It expands the mixed measurement-plus-parameter
population to preserve five independent parameter-continuation training roots
after the fixed validation/test family floors, while keeping the expected
optimizer-visible train-plus-validation rows inside the launcher budget. The
regenerated artifact, not this estimate, is authoritative.

The strict audit quarantines an episode unless its claimed outcome is supported
by the hidden physical truth. Every accepted correction must name an exact
same-family truth target: a grouped measurement correction may not include a
healthy index, and parameter and topology targets remain distinct even on the
same branch. For every terminal outcome, including operator escalation, each
accepted target must also be no farther from clean truth than it was at reset;
missing or malformed initial/final/clean evidence fails closed. A `resolved`
episode additionally requires the active physical store payload, zero faults
in the independently derived remaining-truth ledger, preservation of healthy
measurements and all non-target case fields, and final target
measurements/case fields matching clean truth within their separately declared
tolerances (topology status is exact). A caller-supplied remaining ledger is
optional, but an incomplete or false ledger is rejected and a complete ledger
must agree with the derived count.

Harmonic, HIF, and three-phase-unbalance explanations must match both the true
family and the declared localization tolerance; unbalance may declare an
explicit top-k localization allowance. The only allowed reason-bearing
`not_applicable` declaration is the final fundamental-measurement comparison
used by explanation-only waveform scenarios, because diagnosis does not
rewrite that snapshot. That waiver requires the generator's explicit
`explanation_only_diagnostic_localization_v1` contract, a pure harmonic, HIF,
or three-phase-unbalance root, matching diagnostic truth and localization,
and no correction truth or accepted correction. Accepted-target correctness
and non-regression,
remaining faults, healthy measurements, healthy case components, diagnostic
localization, and final case evidence are never waivable. N/A does not remove
a fault from the derived remaining ledger, so it cannot turn an unlocalized
diagnostic fault into a resolution.

Terminality is not synonymous with successful recovery. The state classes
`terminal_resolved` and `terminal_operator_escalation` are separate, and
`terminal_scenario_matrix` records their counts and physical-root IDs by
family. A verified operator handoff is an auditable safe outcome, but it does
not count as resolution. Release policy v3 reports raw resolution and total
escalation separately, while gating per-family minimum roots, audited
completion, and unqualified escalation. A post-correction handoff qualifies
only when its final observable action, accepted-correction ledger, active
state ID/hash, and controller marker agree and a separate strict offline audit
proves complete target repair, tolerance compliance, and healthy-component
preservation. Missing, conflicting, partial, HIF, or generic handoff evidence
fails closed as unqualified escalation. `measurement+parameter` and
`measurement+topology` each require at least 22 roots, at least 95% audited
completion, and at most 5% unqualified escalation. The 20-root pure
`multi_measurement` family is currently an audited safety/handoff family with
a 0% audited-completion floor and a 100% unqualified-escalation ceiling. This
is an explicit non-claim of autonomous
multi-meter recovery: after a verified partial meter commit, an unavailable or
inconclusive same-state branch route cannot safely authorize another meter
correction. Every such handoff must still be terminal, retain accepted targets,
avoid healthy-component corruption, and record zero false commits,
finalizations, or rollbacks. HIF requires 17 roots and measurement+HIF requires
two roots, both with an explicit unqualified-handoff allowance. The remaining
direct BC0 families require full audited completion and no unqualified
escalation. HIF or partial multi-measurement handoff must not be reported as
general recovery success.

`aggregate.manifest.json` persists each strict audit, the separate private
completion assessment, a non-private final transition/store anchor, and a
zero-false-commit/rollback/finalize/loop lifecycle record. Qualification is
recomputed from those bindings with one vote per physical root; missing,
duplicated, mismatched, or failed evidence blocks release. The manifest hash is
recorded beside the JSONL hashes in generation provenance and is mandatory at
downstream DAgger-1 D0 source gates. Packed handoff artifacts must additionally
retain their externally published archive checksum.

Splits are assigned before descendants are generated and group every row by
`physical_root_fingerprint`. The deterministic split is stratified by network
case, family combination, error cardinality, and source tier, with validation
and test root floors for critical families; the split audit fails closed on
root overlap or coverage deficits. `aggregate.raw.jsonl` is the immutable
eligible natural population. `aggregate.validation.jsonl` and
`aggregate.test.jsonl` preserve their natural held-out distributions.
`aggregate.train_view.jsonl` is the only balanced view: it is deterministically
sampled from natural train rows across state class, target tool/category,
scenario family, cardinality, terminal outcome, and physical root, with bounded
duplication and low-cost-margin exclusions. Balancing never rewrites or
resamples a held-out split.

Release realizability is evaluated on the immutable natural aggregate, not
only on the balanced training view. Exact teacher conflicts must be zero, and
the approximate audit must have real nearest-neighbor and local-perturbation
comparison coverage, bounded disagreement, and cost-margin coverage for
multi-action states. The same approximate gates run separately by scenario
family and by `state_class` decision stage; an empty comparison set is not a
pass. The balanced training view is audited independently as an additional
training-input gate.

`dagger/evaluator.py` rolls policies out on fixed scenario suites. Scenario
schema v1 separates `execution`, the only fields that reach reset, from
`audit`, `grouping`, and the canonical family/cardinality/case/split/source/root
identity; partial, case-variant, ambiguous, or malformed envelopes fail closed.
Offline cost scoring receives copied values but no live environment. The
evaluator reports physical correctness, resolution versus escalation,
healthy-component corruption, false commit/rollback/finalization, partial-fix
retention, invalid-action recovery, loops, WLS and specialized-tool use, and
tool regret, grouped by suite, family, cardinality, case, split, source tier,
and physical root.

`python scripts/build_bc0_evaluation_suite.py --check` deterministically
reconstructs the suite from tracked inputs. BC0 freezes the ten release-policy
families represented in the default aggregate; excluding three-phase unbalance
and telemetry-no-disturbance is a scope decision, not a statement that every
included family has a correction tool. Seed `20260734` controls evaluation-suite
generation order only. Aggregate generation, closed-loop episodes, and suite
fingerprinting use `20260719`; different seeds do not establish independence.
The builder fails before reading suite inputs unless it is running on Python
3.12.x with `numpy==2.3.5`, `scipy==1.16.3`, `PYPOWER==5.1.19`,
`fastmcp==2.12.4`, `OpenDSSDirect.py==0.9.4`, `dss-python==0.15.7`, and
`dss-python-backend==0.14.5`. The OpenDSS pins are part of the builder
contract because aggregate HIF diagnostics execute in this same environment;
an unavailable solver is an infrastructure failure, not negative diagnostic
evidence. The full Python patch version and package versions are reported as
rebuild provenance, but changing that report cannot make `--check` accept
different suite bytes. Development interpreters may run the unit tests, but
cannot build or bless the frozen release artifact.

Direct PYPOWER topology synthesis uses the
`bc0_synthesized_measurement_decimal12_half_even_v1` persistence contract.
Native telemetry first passes the anomaly and corrected-case physics gates;
each finite value is then projected to a `1e-12` decimal lattice with half-even
rounding, signed zero is normalized, and the projected vector passes the same
gates again before scenario materialization. Both round-0 aggregate and
evaluation-suite physical-v3 fingerprints therefore consume the same
platform-stable emitted telemetry; the fingerprint algorithm itself remains
exact and unchanged.

Shared tabular sources are separated before sampling by
`sha256_physical_content_modulo_v1`: bucket 0 of 5 is evaluation, while buckets
1--4 are training. IDs and path aliases are excluded from the physical-content
digest so renamed duplicates stay together. This boundary covers shared
no-error, measurement, multi-measurement, parameter, harmonic, and
measurement+parameter sources. Evaluation HIF uses the curated 17-root
single-scan corpus; training HIF uses the independently generated, QA-passing
17-by-20 diverse multiscan corpus and its tracked QA files. Synthetic topology
families remain protected by the final physical-v3-root and scenario-ID overlap
gate. Aggregate provenance fails release eligibility on any overlap, duplicate,
missing identity, untracked input, or changed suite binding.

`CandidateQualityOracle(mode="synthetic")` requires hidden truth.
`mode="deployment"` ignores it and relies on observable WLS/physics evidence.
The `verifier` package provides deterministic rules plus a structured numerical
model; deterministic safety rules remain authoritative for final acceptance.

The DAgger collector records complete `(s_t, a_t, o_{t+1}, s_{t+1})`
transitions, catches policy/JSON failures, uses updated history for next-state
labels, constructs a deterministic balanced training view without mutating the
natural aggregate, and selects the best validation checkpoint. Counterfactual
recovery generation and top-L AggreVaTe-lite ranking both use isolated
environment clones.
Branch collaborators must therefore be stateless functions or deepcopyable
callable objects. Functions that close over mutable state and non-copyable
solver clients are rejected before branch execution; integrations should wrap
such clients in an explicitly cloneable adapter or supply an external branch
factory to the ranker.

Chat SFT export now emits the full JSON tool schema on every row, keeps tool
arguments dictionary-valued, aliases controller identifiers in the model view,
stores the reverse bindings only in metadata, and retains one bounded history
window. `LocalAliasPolicyAdapter` applies the same view at inference and binds
generated aliases back to episode-local controller IDs.

`dagger/protocol_bridge.py` maps the controller macro surface onto the
canonical power-tool protocol from `trace_protocol.CANONICAL_POWER_TOOLS`
(`wls_from_path`/`case_path`), so DAgger rows can share one model-visible tool
surface with the production SFT corpus, including the harmonic, HSE,
three-phase NLM, and HIF estimator schemas. `examples_to_chat_sft(...,
protocol="canonical")` exports canonical targets (correction values are
dropped; the model is supervised on target selection only) and
`LocalAliasPolicyAdapter(..., protocol="canonical")` converts generated
canonical calls back before alias binding. Canonical is the default for both
deployment export and inference. Historical controller-protocol fixtures must
request `protocol="controller"` explicitly.
Reverse-mapped `correct_*_from_path` calls carry targets without values and
therefore require deployment correction providers that hydrate values before
they can execute; canonical-only diagnostics pass through and no-op until
their executors are integrated.

`production_dataset_mode=True` fails closed unless WLS, all three context
providers, and all three correction executors declare production provenance or
are explicitly approved deterministic pilot adapters. Production labels for
domain context, correction, commit, rollback, and finalization require the
corresponding observable evidence. Bounded context findings and exact
`supported_corrections` remain model-visible, and production correction targets
must match them exactly. Round-0 generation performs the grouped split before
chat export and records the strict truth audit, native/chat schema checks,
exact and approximate realizability reports, target-aware state-class audit,
terminal family matrix, and generation provenance in its preflight artifacts.

## Running experiments

The research pipeline runs from the Slurm cell in
`research/hpc/full_pipeline_20260907/`; its README lists the stages, the
corpora, and every recorded run.

| Stage | Entry point |
|---|---|
| Expert aggregate (D0) | `python -m psse_env.examples.generate_round0_aggregate` |
| DAgger scenario suites | `research/hpc/full_pipeline_20260907/build_suite.py` |
| BC0 and DAgger-round training | `python -m psse_env.sft research-train` |
| Collection and paired evaluation | `scripts/run_dagger_research.py` |
| Round and pipeline summaries | `research/hpc/full_pipeline_20260907/summarize.py` |

The release-training CLI, the DAgger-1 collection and builders, the study
manifests, and the root-level Slurm launchers that earlier versions of this
file documented were removed on 2026-09-28; they remain in the git history.
