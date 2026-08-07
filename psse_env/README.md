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

## Round-0 BC0 release path

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
quotas and thresholds before claiming either one. Every selected scenario
passes its physical validation gate, and scenario IDs are opaque hashes;
family, cardinality, network case, and source tier remain audit/split metadata
rather than policy-visible hints.

`examples/generate_round0_aggregate.py` is expert-only collection at
`β=1.0`. It produces a candidate BC0 behavioral-cloning corpus, not a DAgger
iteration. Before the environment, policy, or online expert receives a root,
the collector deep-copies it and removes `true_*`, `clean_*`, `hidden_truth`,
and oracle action hints. The original scenario is retained outside the online
trajectory and is supplied only after termination to the strict offline audit
in `dagger/release_audit.py`; audit truth and audit results are never merged
into model observations or SFT targets.

BC0 uses the machine-readable supervision policy
`bc0_observable_sequential_v1`. At each expert-controlled state, the policy
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

The release regeneration uses the tracked ten-family row-budget plan rather
than `--scale 2`, which would incorrectly request 34 HIF roots from the
17-root training inventory:

```bash
python -m psse_env.examples.generate_round0_aggregate \
  --plan data/round0_plan_20260719.json \
  --output-dir data/round0_aggregate_release
```

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
not count as resolution. Release policy therefore enforces per-family minimum
root counts, resolution floors, and escalation ceilings; nonterminal,
quarantined, or unknown terminal outcomes fail. `measurement+parameter` and
`measurement+topology` each require at least 20 roots, at least 95% resolution,
and at most 5% escalation. The 20-root pure `multi_measurement` family is
currently an audited safety/handoff family with a 0% resolution floor and a
100% escalation ceiling. This is an explicit non-claim of autonomous
multi-meter recovery: after a verified partial meter commit, an unavailable or
inconclusive same-state branch route cannot safely authorize another meter
correction. Every such handoff must still be terminal, retain accepted targets,
avoid healthy-component corruption, and record zero false commits,
finalizations, or rollbacks. HIF requires 17 roots and measurement+HIF requires
two roots, both with an explicit audited handoff allowance. The remaining
direct BC0 families require full resolution and no escalation. Audited HIF or
multi-measurement handoff must not be reported as general recovery success.

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

Checkpoint decisions must also persist a reproducible closed-loop evaluator
report on fixed scenario suites. Development runs may retain legacy flat
scenarios behind normalized recursive truth stripping. Release runs require
scenario schema v1 with exactly `scenario_schema_version`, `execution`, `audit`,
and `grouping`: only a positive allowlist of execution and deployment-metadata
fields reaches reset, while audit truth, labels, and the required canonical
family/cardinality/case/split/source/root identity remain outside execution.
Partial, case-variant, ambiguous, or malformed envelopes fail closed. Release
scenarios also reject the scripted-transition and injected-physical-state
fields reserved for development test adapters. Every evaluation records this
schema-v1 check, and both the artifact writer and release gate require the exact
passing attestation before reporting release eligibility. Offline
cost scoring receives copied values but no live environment, and custom cost or
physical-audit callbacks are development-only; their use makes an artifact
release-ineligible. Release execution also requires every actual environment to attest
`production_dataset_mode=true` and
`candidate_quality_oracle.mode="deployment"`. Every instantiated policy must
also expose an exact `release_policy_identity` matching the requested explicit
policy identity or immutable model ID/revision; the evaluator persists that
attestation for every episode. The evaluator reports physical
correctness, resolution versus escalation, healthy-component corruption,
false commit/rollback/finalization, partial-fix retention, invalid-action
recovery, loops, WLS and specialized-tool use, and tool regret, grouped by
suite, family, cardinality, case, split, source tier, and physical root.

`python -m psse_env.dagger.validate_evaluation` implements the content-pinned
v2 contract in `dagger/bc0_evaluation_policy.json`. It recomputes identity
hashes, binds the artifact to the current clean commit and frozen suite,
enforces strict audit v3, exact-matches schema-v1 evaluator configuration and approved
factory identities, and checks hard safety, terminality, loop, invalid-call,
and per-family root/outcome constraints before any scalar score is considered.
The packaged policy pins the repository-tracked schema-v1 suite and the exact
reviewed environment, observable-expert, Gemma-policy, and case-loader factory
module. The suite contains 21 globally unique physical roots in each of standard
success, forced-error recovery, partial-success retention, and invalid-action
recovery, plus 31 in the efficiency suite, for 115 globally unique roots.  The
efficiency suite deliberately mixes the diagnostic families (HIF,
measurement+HIF, harmonic) with core state-estimation families (measurement,
parameter, topology, measurement+parameter, measurement+topology) so
checkpoint promotion bounds tool-call efficiency for mixed-error recovery,
not only diagnostic escalation. The environment factory requires production dataset
mode, a deployment candidate oracle, and the 24-step protocol. The expert
factory consumes policy-safe observations only and exposes identity
`bc0-observable-expert-v1`. The Gemma factory loads only the exact local
`unsloth/gemma-4-31B-it@8a796db4df380b178065ed910849477ff0e99c87`
snapshot, verifies its byte manifest, and content-addresses any PEFT adapter.
The case loader resolves repository-root paths deterministically through the
production MATPOWER parser.

The freeze is atomic: suite artifact, builder, factories, policy/family matrix,
tests, and HIF training/QA inputs land in one commit. At that exact clean
commit, run `scripts/build_bc0_evaluation_suite.py --check`, recompute the suite
SHA-256 and fingerprint manifest, pin every factory `import_spec` and source
SHA-256, set policy status to `pinned`, and only then regenerate aggregate and
evaluation evidence. Any later edit to a frozen input invalidates that evidence.

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

Roles have different blocking semantics. The expert baseline must pass both
evidence and performance. The base-model baseline records the same performance
failures, but only incomplete identity or evaluation evidence blocks BC0
training; a weak but reproducibly evaluated base is never labeled
release-qualified. Checkpoint promotion again requires the full absolute gate
and an exact paired base-reference comparison: evaluator, environment,
model-policy factory, case-loader, configuration, and episode identities must match, and no paired
episode may regress in safety ordinal or invalid-action count. Aggregate gains
and scalar scores cannot hide a regressed root.

These artifacts establish reproducibility and internal consistency, not a
cryptographic attestation that the named code was the code executed. Trusted
runner isolation, signing, and custody controls remain deployment concerns.
Before deployment promotion, publish threshold sensitivity for the hard and
per-family limits, including margins and the roots whose decisions change
under predeclared nearby thresholds; sensitivity analysis supplements and
never rewrites the pinned pass/fail policy. These reports remain required
release evidence, not substitutes for corpus preflight or strict episode
auditing. The current BC0 scope permits audited HIF handoff; it validates safe
HIF escalation, not autonomous HIF localization or repair.

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

Current release decision: the evaluated BC0-v1 adapter is **NO-GO for
promotion and NO-GO for additional epochs on D0 alone**, but it is retained as
a learner seed for the bounded DAgger-1 collection below. The exact
same-state correction guard, deterministic-failure breaker, private-oracle
truth separation, fresh-root builder, and recovery-label gates must all come
from one clean reviewed commit. Historical pilot and round-0 artifacts are
evidence inputs only; they are not release-eligible by inheritance. A repaired
checkpoint must still pass fixed-suite expert/base prerequisites, exact
processor/template/mask and optimizer gates, and its own paired closed-loop
promotion gate. No expert-only `β=1.0` corpus, failed-promotion learner seed,
or diagnostic replay should be labeled as promoted DAgger evidence.

HPC release prerequisites are H200, H100, or a high-memory RTX 6000, Python
3.12, the exact `requirements-sft.txt` versions, a passing `pip check` and
launcher version audit, and a complete local pinned Gemma snapshot. The Slurm
feature permits `rtx6000`, but the runtime gate accepts that family only when
exactly one GPU is visible, BF16 is supported, and total memory is at least
90,000 MiB; this excludes 48-GB variants from the 31B release path. Release
evaluation cannot download or fall back to another revision. SFT training loads
through the same byte-verified snapshot path as release evaluation: the trainer
refuses any model identity other than the pinned base, verifies the snapshot
manifest before and after loading, accepts only the Gemma 4 conditional model
class with no `AutoModelForCausalLM` fallback, writes a durable
`base_snapshot_attestation.json` beside the training output, and normalizes
the saved adapter's `base_model_name_or_path` back to the pinned Hub ID so the
checkpoint gate can promote it. The launcher additionally asserts Python 3.12,
`torch 2.10.0+cu128`, and an approved release accelerator, and forbids
`ALLOW_DOWNLOAD=1` outside the dataset-only gate stage. Both launchers attach
the Slurm comment `preemption=yes;requeue=true`; this is scheduler metadata and
does not itself add signal-driven checkpointing or automatic resume to the BC0
trainer. Smoke-stage timing must justify the 24-hour allocation for the
two-epoch 31B run. The launcher accepts only a legitimate audited aggregate
within `ROWS_MIN=1024` and `ROWS_MAX=4096`; if a valid aggregate exceeds the
upper bound, raise `ROWS_MAX` rather than trimming rows. Those bounds apply to
optimizer-visible train plus validation rows; the gate still audits the
held-out test split and offsets its total-row check by the test count so every
stage enforces the same training-corpus size.

Both HPC launchers require the reviewed freeze commit as an external input;
they never bless the checkout's current `HEAD` on their own.  The release
evaluator also fixes all outputs below `artifacts/evaluations/`, rejects path
aliases that could overwrite the persisted base artifact, and requires
checkpoint evaluation to point at the final adapter directory (`output/lora`),
not a Trainer checkpoint containing optimizer state.  Submit the base and one
content-addressed checkpoint evaluation around the SFT stages as follows:

```bash
FREEZE_COMMIT=$(git rev-parse HEAD)
sbatch --export=ALL,REVIEWED_SOURCE_COMMIT="$FREEZE_COMMIT",EVALUATION_MODE=expert \
  submit_dagger_release_eval.sh

sbatch --export=ALL,REVIEWED_SOURCE_COMMIT="$FREEZE_COMMIT",EVALUATION_MODE=base \
  submit_dagger_release_eval.sh
```

After both baseline gates pass, run `gate -> one-batch -> tiny-overfit ->
round0` in that order. Only after `round0` produces `output/lora` should the
checkpoint evaluation run:

```bash
sbatch --export=ALL,REVIEWED_SOURCE_COMMIT="$FREEZE_COMMIT",EVALUATION_MODE=checkpoint,CHECKPOINT_PATH=/absolute/output/lora \
  submit_dagger_release_eval.sh
```

Pass the same `REVIEWED_SOURCE_COMMIT` to every
`submit_dagger_sft_round0.sh` stage. Slurm logs are written at the repository
root so submission does not depend on a pre-existing untracked log directory.
`STAGE=gate` writes the durable `PROCESSOR_GATE_REPORT` (default
`$OUTPUT_DIR/gate_report.json`); every stage must share that path or the same
`OUTPUT_DIR`. `STAGE=round0` revalidates that report against the exact source,
model revision, maximum length, and train/validation/test bytes and requires
that it used `AutoProcessor`. The checkpoint evaluation is followed by
`STAGE=checkpoint-gate`; it is not part of the baseline submission group.

For a reviewed Round-1 aggregate, warm-start from the exact content-addressed
BC0 learner-seed LoRA tree into a fresh, non-overlapping output directory. The
seed remains release-ineligible; its role here is only initialization. The launcher independently
validates the adapter contents and its 64-hex tree identity, loads it through
PEFT with `is_trainable=True`, restores its exact pre-smoke trainable weights
before TRL starts, and writes `initial_adapter_attestation.json`.
`STAGE=round1` also refuses any revision that differs from both the D1
collection manifest and the aggregate's learner-seed provenance. Round 1
defaults to one epoch at `3e-5`; override those values only as an explicit
experiment change:

```bash
export INITIAL_ADAPTER_PATH=/absolute/round0/output/lora
export INITIAL_ADAPTER_REVISION=<64-hex-checkpoint-tree-sha256>
export OUTPUT_DIR=/absolute/fresh/round1/output

sbatch \
  --export="ALL,STAGE=round1,REVIEWED_SOURCE_COMMIT=$FREEZE_COMMIT" \
  submit_dagger_sft_round0.sh
```

### Targeted DAgger-1 recovery flow

Run this flow only after committing the runtime/data changes and regenerating
the current-source D0 aggregate. The scenario builder uses the train source
partition at the release parameter-routing threshold `1.0`, over-generates
before filtering, and rejects every physical root already present in D0 or the
frozen 115-root evaluation suite:

```bash
export FREEZE_COMMIT=$(git rev-parse HEAD)
export D0_DIR=data/round0_aggregate_release
export D1_DIR=data/dagger1_${FREEZE_COMMIT:0:12}
mkdir -p "$D1_DIR"

python scripts/build_dagger1_scenarios.py \
  --d0-aggregate-dir "$D0_DIR" \
  --output "$D1_DIR/scenarios.json" \
  --generator-report "$D1_DIR/scenario_generator_report.json" \
  --seed 20260720

python scripts/build_dagger1_development_holdout.py \
  --d0-aggregate-dir "$D0_DIR" \
  --d1-training-scenarios "$D1_DIR/scenarios.json" \
  --d1-training-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --output "$D1_DIR/development_holdout.json" \
  --generator-report "$D1_DIR/development_holdout.generator.json" \
  --seed 20260721
```

The second command freezes 30 additional diagnostic development roots in a
12/12/6 family split; the first command reserves the complementary 120
training roots in a 48/48/24 split. They are byte-bound and disjoint from D0,
D1 training, and the frozen 115-root promotion suite, but are explicitly
ineligible for SFT and promotion evidence.

Use the failed-promotion adapter only as the learner. The diagnostic pass is
learner-only and always training-ineligible; the mixed pass must produce
300--600 eligible learner-recovery rows, pass the offline physical-truth target
audit, provide at least five distinct roots in every targeted state cell, and
meet the 10-root central / 5-root remaining recovery-stratum floors:

For D1, both the current teacher target and the stored next-action inventory
are functions of `PolicyObservation` only. `OracleState` is constructed after
the target and rank-one proof are fixed, solely for quarantine and transition
auditing. Teacher calls receive only the exact bounded `history_window` carried
by that observation; the collector's fuller post-transition audit history is
never an auxiliary teacher input.

```bash
export LEARNER_ADAPTER=/absolute/path/to/bc0-v1/lora
export LEARNER_REVISION=$(python -c \
  'import sys; from psse_env.dagger.release_factories import inspect_release_checkpoint; print(inspect_release_checkpoint(sys.argv[1])["tree_sha256"])' \
  "$LEARNER_ADAPTER")

python scripts/collect_dagger1_recovery.py \
  --input "$D1_DIR/scenarios.json" \
  --scenario-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --scenario-generator-report "$D1_DIR/scenario_generator_report.json" \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --d0-aggregate-dir "$D0_DIR" \
  --output "$D1_DIR/diagnostic_beta0.jsonl" \
  --all-output "$D1_DIR/diagnostic_beta0.all.jsonl" \
  --model-id "$LEARNER_ADAPTER" \
  --model-revision "$LEARNER_REVISION" \
  --collection-pass diagnostic --beta 0.0

python scripts/collect_dagger1_recovery.py \
  --input "$D1_DIR/scenarios.json" \
  --scenario-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --scenario-generator-report "$D1_DIR/scenario_generator_report.json" \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --d0-aggregate-dir "$D0_DIR" \
  --output "$D1_DIR/training_beta025.jsonl" \
  --all-output "$D1_DIR/training_beta025.all.jsonl" \
  --model-id "$LEARNER_ADAPTER" \
  --model-revision "$LEARNER_REVISION" \
  --collection-pass training --beta 0.25 \
  --failed-collection-dir "$D1_DIR/training_beta025.failed-collection" \
  --require-recommended-target

python scripts/build_dagger1_training_aggregate.py \
  --d0-aggregate-dir "$D0_DIR" \
  --d1 "$D1_DIR/training_beta025.jsonl" \
  --d1-manifest "$D1_DIR/training_beta025.jsonl.manifest.json" \
  --output-dir data/round1_aggregate_release \
  --d1-share 0.25
```

If the strict mixed-policy gate fails, the command exits nonzero and leaves
`training_beta025.jsonl`, its all-row ledger, and its production manifest
absent. Instead it atomically publishes the explicitly named
`training_beta025.failed-collection/` directory with checksummed,
training-ineligible candidate rows, all visited rows, and structured gate
reports. Use that bundle to diagnose the failure; never pass it to the Round-1
aggregate builder. Both successful production outputs and failed diagnostic
bundles are write-once, so choose a new attempt path for each rerun.

The collection manifest reports the default Round-1 replay capacity and the
largest feasible total view under the duplicate and per-root caps. If the
default natural-union size is infeasible, collect more D1 rows or pass an
explicit smaller `--size`; never relax the 20--30% source-share band silently.
The final builder independently reruns the D1 row audits and exact/approximate
teacher-realizability gates, preserves `aggregate.raw.jsonl`,
`aggregate.d0.raw.jsonl`, `aggregate.d1.raw.jsonl`, and
`aggregate.train_view.raw.jsonl`, and hashes every semantic report into
generation provenance.

Regenerate the expert and base artifacts at `FREEZE_COMMIT`, then run the
Round-1 chain against `data/round1_aggregate_release` in this order:
`gate -> one-batch -> targeted-tiny-overfit -> round1 -> checkpoint evaluation
-> checkpoint-gate`. Submit the data-only `gate` without
`INITIAL_ADAPTER_PATH`/`INITIAL_ADAPTER_REVISION`; export the learner-seed
identity for `one-batch`, `targeted-tiny-overfit`, and `round1`. The targeted
stage selects five distinct recovery cases,
sweeps diagnostic learning rates `1e-4`, `3e-4`, and `1e-3`, and requires exact
generated tools and arguments. These sweep rates do not alter the Round-1
training default of `3e-5` for one epoch.

After each candidate training configuration, evaluate its adapter on
`$D1_DIR/development_holdout.json` with the diagnostic-only evaluator before
using the frozen suite. This artifact is for model selection only:

```bash
python scripts/evaluate_checkpoint_diagnostic.py \
  --input "$D1_DIR/development_holdout.json" \
  --output "$D1_DIR/development_${CANDIDATE_REVISION}.evaluation.json" \
  --env-factory psse_env.dagger.release_factories:production_environment_factory \
  --policy-factory psse_env.dagger.release_factories:gemma_release_policy_factory \
  --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
  --model-id "$CANDIDATE_ADAPTER" \
  --model-revision "$CANDIDATE_REVISION" \
  --minimum-roots-per-suite 30 --max-steps 24 --seed 20260721
```

The relaxed accelerator policy remains unchanged: each artifact may use any
approved H200, H100, or high-memory RTX 6000. Same-class reruns are useful for
the cleanest causal comparison when capacity permits, but are not a hard gate.

The current diagnostic evaluator binds the exact holdout roots and reports
closed-loop outcomes, but its persisted trace retains only an observation hash
and the learner action. It therefore cannot yet certify coverage of the
expert-defined recovery strata, which require the policy-safe observation and
the expert's fixed rank-one target. The holdout manifest marks stratum-qualified
model selection false until a diagnostic-only teacher-opportunity trace and
post-evaluation coverage audit are implemented; do not claim recovery-stratum
generalization from family counts alone.

To cheaply replay the reviewed 13 failures after the runtime patch, first
build a separate suite, then use the diagnostic-only evaluator and audit it:

```bash
python scripts/build_checkpoint_diagnostic_suite.py \
  --artifact /path/to/failed_checkpoint_evaluation.json \
  --frozen-suite psse_env/dagger/suites/bc0_eval_suite_v1.json \
  --output "$D1_DIR/checkpoint_failure_suite.json" \
  --expected-scenarios 13

python scripts/evaluate_checkpoint_diagnostic.py \
  --input "$D1_DIR/checkpoint_failure_suite.json" \
  --output "$D1_DIR/checkpoint_failure_evaluation.json" \
  --env-factory psse_env.dagger.release_factories:production_environment_factory \
  --policy-factory psse_env.dagger.release_factories:gemma_release_policy_factory \
  --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
  --model-id "$LEARNER_ADAPTER" \
  --model-revision "$LEARNER_REVISION" \
  --max-steps 24 --seed 20260719

python scripts/validate_checkpoint_diagnostic.py \
  --artifact "$D1_DIR/checkpoint_failure_evaluation.json" \
  --diagnostic-suite "$D1_DIR/checkpoint_failure_suite.json" \
  --output "$D1_DIR/checkpoint_failure_evaluation.audit.json"
```

Diagnostic artifacts use a distinct artifact type and are irreversibly marked
release- and training-ineligible. Approved H200, H100, and high-memory RTX 6000
evaluations remain independently hardware-attested; base and checkpoint jobs
do not need to land on the same approved accelerator class.

### Optional W&B monitoring

Weights & Biases monitoring is disabled by default and activates only for
`STAGE=round0` or `STAGE=round1`; the data gate and smoke stages remain local
evidence checks.
It is safe to include `ENABLE_WANDB=1` in the shared export used by the whole
staged chain: earlier stages log that monitoring is inactive and do not import
or initialize W&B.
Install the separately pinned optional dependency once on a login node, then
authenticate interactively there so no API key is placed in an `sbatch`
command or job log:

```bash
INSTALL_WANDB=1 bash setup_unsloth_env.sh
conda activate /scratch/yx3882/.conda/envs/unsloth_sft
wandb login
```

Add W&B variables to the existing reviewed round-0 submission. Export tags in
the shell rather than embedding their comma-separated value inside Slurm's
comma-delimited `--export` argument:

```bash
export WANDB_PROJECT=psse-agent-bc0
export WANDB_ENTITY=your-wandb-user-or-team
export WANDB_RUN_GROUP=bc0-round0
export WANDB_TAGS=bc0,round0,gemma4-31b
export WANDB_JOB_TYPE=bc0-round0-sft

sbatch \
  --export="ALL,STAGE=round0,ENABLE_WANDB=1,REVIEWED_SOURCE_COMMIT=$FREEZE_COMMIT" \
  submit_dagger_sft_round0.sh
```

The same opt-in works for Round 1; stage-specific default group, tags, job type,
run ID, and name use `round1`/`r1` unless explicitly overridden.

The launcher defaults to a stable run ID and name from the reviewed commit plus
the Slurm job ID, sets `WANDB_RESUME=allow` for a requeued job, and disables
model uploads and parameter watching. Set the same `WANDB_RUN_ID` explicitly
when a newly submitted replacement job should resume an earlier run; a custom
`WANDB_NAME` is also preserved. W&B run, cache, data, configuration, and
artifact directories default below `/scratch/$USER/wandb`; override
`WANDB_SCRATCH_ROOT` if a different scratch allocation is required.

For a compute node without outbound network access, set
`WANDB_MODE=offline` before submitting with `--export=ALL`. The complete run
will remain under the scratch W&B directory and can later be uploaded from a
login or data-transfer node with `wandb sync /path/to/offline-run-*`.

Run the dedicated gate with:

```bash
uv run --with 'pytest>=8,<9' pytest -q \
  test_psse_dagger_scaffold.py \
  psse_env/verifier/test_hardening.py \
  psse_env/dagger/test_evaluator.py \
  psse_env/dagger/test_evaluation_gate.py \
  psse_env/dagger/test_collect_dagger1.py \
  psse_env/dagger/test_dagger1_builders.py \
  psse_env/dagger/test_dagger1_development_holdout.py \
  psse_env/dagger/test_dagger1_semantic_audit.py \
  psse_env/dagger/test_diagnostic_suite.py \
  psse_env/dagger/test_offline_teacher_target_audit.py \
  psse_env/dagger/test_review_regressions.py \
  psse_env/dagger/test_training_view.py \
  psse_env/dagger/test_suite_builder.py \
  psse_env/dagger/test_release_factories.py \
  psse_env/dagger/test_release_eval_launcher.py \
  psse_env/dagger/test_sft_export.py \
  psse_env/providers/test_scenario_generator.py \
  psse_env/examples/test_generate_round0_aggregate.py \
  psse_env/test_production_mode.py \
  psse_env/sft/tests/test_gates.py \
  psse_env/sft/tests/test_release_hardware.py \
  psse_env/sft/tests/test_wandb_launcher.py \
  psse_env/sft/tests/test_wandb_training.py \
  psse_env/sft/tests/test_targeted_tiny_overfit.py \
  psse_env/sft/tests/test_warm_start_training.py
```

See `BASELINE.md`, `TEST_MANIFEST.md`, and `EXPERIMENT_PROTOCOL.md` for the
checkpoint, test grouping, reproducible fixtures, and experiment gates.
