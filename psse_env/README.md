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
Candidate verification derives observable `target_fixed`/
`remaining_fault_count` evidence from the candidate solve alone.

The five specialized diagnostics — `get_harmonic_context`,
`run_hse_from_path`, `run_three_phase_nlm_from_path`, and both HIF
estimators — are first-class macro actions sharing their canonical deployment
names. The process gate treats them as read-only evidence actions (legal on
the active state, or on an INCONCLUSIVE candidate), and the environment
dispatches them through `evidence_providers` with the full action so bounded
estimator arguments (candidate branch, phase, grid options) reach the tool.
The deployment bundle wraps the real HSE, three-phase NLM, and HIF estimation
stacks; runtime side data comes from state metadata
(`harmonic_measurements`/`harmonic_orders`, `nlm_diagnostic` or OpenDSS model
dirs, `hif_runtime`, `hif_scan_window`) and each tool fails closed as a
collectable no-op when its data is absent. The protocol bridge maps
`state_id` to `case_path` (or `scan_window_path` for the multi-scan
estimator) so exported targets and generated calls stay canonical.

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
canonical calls back before alias binding. The default remains
`protocol="controller"`, so the validated 90-row pilot is unchanged.
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
