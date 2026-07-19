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

The five specialized diagnostics — `get_harmonic_context`,
`run_hse_from_path`, `run_three_phase_nlm_from_path`, and both HIF
estimators — are first-class macro actions sharing their canonical deployment
names. The process gate treats them as read-only evidence actions (legal on
the active state, or on an INCONCLUSIVE candidate), and the environment
dispatches them through `evidence_providers` with the full action so bounded
estimator arguments (candidate branch, phase, grid options) reach the tool.
The deployment bundle wraps the real HSE, three-phase NLM, and HIF estimation
stacks; runtime side data comes from state metadata
(`harmonic_measurements`/`harmonic_orders`, `three_phase_voltages`,
`nlm_diagnostic` or OpenDSS model dirs, `hif_runtime`, `hif_scan_window`) and each tool fails closed as a
collectable no-op when its data is absent. The protocol bridge maps
`state_id` to `case_path` (or `scan_window_path` for the multi-scan
estimator) so exported targets and generated calls stay canonical.

`PolicyObservation.available_evidence` lists which telemetry channels exist
on the active state (deployment-observable operator knowledge; channel
contents stay out of the observation). `DiagnosticsExpert` routes on it plus
unresolved-signature markers: harmonic signals escalate
`get_harmonic_context` -> `run_hse_from_path`; pure three-phase-unbalance
signals stop at a VUF/null-gated non-HIF classification; HIF-specific signals escalate
`run_three_phase_nlm_from_path` -> the multi-scan estimator when a persistent
scan window exists, else the single-scan estimator, carrying the NLM top
branch as `candidate_branch_row0`. Privileged fault families and hints are
ignored by this expert, so changing hidden truth while holding the policy
observation fixed cannot change the production target.
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
Two more physical guards close that masking channel: while an unexplained
harmonic/unbalance/HIF sensor signature stands, `run_wls` mints no `wls_*` signatures at
all (the fundamental-frequency solve is unreliable under waveform anomalies),
and `get_topology_context` filters supported status flips that would island
the network (an EMS would never offer that switching action). A candidate
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

`providers/scenario_generator.py` builds the round-0 offline aggregate from
real physics: `Round0ScenarioGenerator` adapts the merged measurement corpus
(single and multi gross outliers, corrupted-parameter cases with multi-scan
data, harmonic and HIF rows with their runtime side channels), synthesizes
topology scenarios via pypower power flows on status-flipped IEEE-14 cases,
and composes measurement overlays on top of other families. Every scenario
passes a physical validation gate (anomalous as observed, clean once the
truth is restored — harmonic/HIF rows are anomalous by nature and gate on
solvability), and scenario IDs are opaque hashes: the family lives only in
the generator manifest, never in policy-visible metadata.
`examples/generate_round0_aggregate.py` drives expert-only collection
(β=1.0) over a family plan, injects bounded per-family counterfactual
recovery branches, audits every episode against hidden truth (masking
commits are quarantined), splits by root scenario, and exports canonical
chat SFT with the native-row, teacher-realizability, and target-aware
audits.

The preflight report treats a terminal decision and a resolved diagnosis as
different outcomes. `terminal_scenario_matrix` records `resolved` and
`operator_escalation` counts and IDs separately. `release_terminal_coverage`
accepts either a resolved episode or an audited, state-bound operator handoff;
`release_resolution_coverage` is true only when every episode resolves.
Nonterminal, quarantined, or terminal episodes with an unknown outcome fail
the release gate.

`CandidateQualityOracle(mode="synthetic")` requires hidden truth.
`mode="deployment"` ignores it and relies on observable WLS/physics evidence.
The `verifier` package provides deterministic rules plus a structured numerical
model; deterministic safety rules remain authoritative for final acceptance.

The DAgger collector records complete `(s_t, a_t, o_{t+1}, s_{t+1})`
transitions, catches policy/JSON failures, uses updated history for next-state
labels, balances aggregate replay by recovery class, and selects the best
validation checkpoint. Counterfactual recovery generation and top-L
AggreVaTe-lite ranking both use isolated environment clones.
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
must match them exactly. The grouped pilot generator splits root
scenarios before chat export and runs native-row, teacher-realizability, and
target-aware replay audits.

Current launch status is deliberately narrower than full training: the 90-row
pilot, exact pinned 31B processor/template/mask gate, and local E2B QLoRA smoke
are approved. Full 31B SFT remains **NO-GO** until the exact 31B checkpoint
passes forward/backward and tiny-overfit gates on suitable HPC hardware and a
short held-out recovery evaluation passes. The bundled pilot contains standard
successful paths, not the required recovery-class coverage. See
`SFT_PILOT_VALIDATION.md`.

Run the dedicated gate with:

```bash
uv run --with 'pytest>=8,<9' pytest -q \
  test_psse_dagger_scaffold.py \
  psse_env/verifier/test_hardening.py \
  psse_env/dagger/test_sft_export.py \
  psse_env/test_production_mode.py \
  psse_env/sft/tests/test_gates.py
```

See `BASELINE.md`, `TEST_MANIFEST.md`, and `EXPERIMENT_PROTOCOL.md` for the
checkpoint, test grouping, reproducible fixtures, and experiment gates.
