# Preregistered multi-seed DAgger study

The version-1 study protocol is the content-addressed JSON file
`psse_env/dagger/studies/dagger_multiseed_study_v1.json`. It is deliberately a
protocol contract, not evidence that training or evaluation has occurred.

## Frozen experiment matrix

The only permitted model variants are:

| Variant | Initialization | Training sources |
| --- | --- | --- |
| `base` | Exact pinned Gemma base | None |
| `bc0` | Exact pinned Gemma base | D0 BC only |
| `natural_dagger` | Same-seed BC0 checkpoint | D0 BC plus natural D1 |
| `natural_dagger_probes` | Same-seed BC0 checkpoint | D0 BC plus natural D1 plus observable probes |

Every trained variant must be materialized for all three preregistered seeds:
`3407`, `3408`, and `3409`. The base model is an untrained reference, so its
evaluation records a null training seed. Adapted comparisons are paired by
training seed; a DAgger seed may not warm-start from a different BC0 seed.

The primary contrast is `natural_dagger_probes - bc0` on multi-error episode
recovery rate. The required probe ablation is
`natural_dagger_probes - natural_dagger`, paired by training seed and evaluated
only on the guaranteed recovery-stress opportunities. Each of the two
probe-targeted recovery-action accuracies must improve by a strictly positive
absolute amount. The unrelated nonregression guardrail is an explicit registry
of unit-interval, higher-is-better diagnostic, physical, and recovery outcome
rates; each may degrade by at most 0.02. Counts and evidence support, safety
counts, action-quality/efficiency summaries, identifiers, and threshold
constants are excluded from that registry because their separately pinned hard
rules govern them. Missing targeted or registry-required evidence fails closed.

Every variant requires three distinct evaluation roles. `development_evaluation`
binds the same exact 30-root development-holdout content hash, provenance ID,
and root-set hash under the diagnostic model-selection-only protocol.
`evaluation` binds the untouched frozen suite and release policy.
`recovery_stress_evaluation` binds a 70-episode, seven-stratum intervention
suite and its generation manifest. Development and frozen are the two primary
recovery/stability scopes; the stress scope is the sole source for the seven
recovery-action thresholds, stress safety counts, and probe ablation. All three
are mandatory and none can substitute for another. Adapted evaluation artifacts additionally
bind the checkpoint receipt ID and require their model revision to equal the
receipt's canonical adapter-tree SHA-256. Base evaluations record both
checkpoint fields as null.

The manifest also pins every numerical objective used by the final comparison,
including diagnostic/physical recovery, named recovery-action accuracy, action
quality and efficiency, single-error non-regression, 5-point seed spread,
2-point per-family material-regression tolerance, and all zero-count safety
constraints. Unsupported or empty evidence fails closed; the comparison may
not silently substitute a default threshold.

## Immutable and external bindings

The Python validator pins the raw JSON bytes. `.gitattributes` fixes the file to
LF on Windows and Linux, so line-ending conversion cannot silently create a
different protocol. It also recomputes the checked-in suite and policy hashes
and verifies that the policy approves that suite, evaluator seed, and step
budget.

The reviewed source commit is intentionally supplied when a run is
materialized. A Git commit cannot contain its own final commit hash without a
self-reference. Each future checkpoint and evaluation must therefore record:

- the immutable study-manifest SHA-256;
- the same externally reviewed clean 40-hex Git commit;
- its exact variant and preregistered training seed (null only for base);
- exact base/checkpoint model identities;
- checkpoint training-view provenance;
- the exact source-gate-authenticated production-D1 quarantine summary,
  candidate/quarantine counts, and report hash (canonical null for BC0); and
- the frozen suite and evaluation-policy hashes; and
- for recovery stress, the exact suite, manifest, provenance, development-parent,
  and root-set hashes plus the fixed episode/root counts.

`validate_study_artifact_binding` enforces those fields before a later
comparison consumes an artifact. This contract does not fabricate any trained
checkpoint or evaluation artifact.

## Bound evaluation invocation

`psse_env.dagger.evaluator` is the sole producer for study evaluation fields.
For the base model it emits the canonical null training seed, receipt ID, and
adapter-tree binding. For `bc0`, `natural_dagger`, and
`natural_dagger_probes`, it validates the complete write-once checkpoint
receipt, requires its variant/source/seed to match the requested arm, resolves
the receipt's adapter path, and independently recomputes the live adapter-tree
SHA-256. A caller cannot provide a receipt ID, tree hash, holdout hash, root-set
hash, provenance ID, or evaluator-contract hash directly.

Development evaluation is one exact protocol: diagnostic-only, canonical tool
protocol, seed `20260721`, 24 steps, the single required
`dagger1_development` suite, minimum suite/episode counts of 1, and a 30-root
floor. The evaluator verifies the holdout bytes against its generation
manifest and generator report and derives the bound hashes. For a trained
candidate:

```bash
python scripts/evaluate_checkpoint_diagnostic.py \
  --input "$D1_DIR/development_holdout.json" \
  --output "artifacts/evaluations/development_${STUDY_VARIANT}_seed${TRAIN_SEED}_${CANDIDATE_REVISION}.json" \
  --env-factory psse_env.dagger.release_factories:production_environment_factory \
  --policy-factory psse_env.dagger.release_factories:gemma_release_policy_factory \
  --case-loader psse_env.dagger.release_factories:deterministic_case_loader \
  --model-id "$CANDIDATE_ADAPTER" \
  --model-revision "$CANDIDATE_REVISION" \
  --study-manifest psse_env/dagger/studies/dagger_multiseed_study_v1.json \
  --study-variant "$STUDY_VARIANT" \
  --reviewed-source-commit "$FREEZE_COMMIT" \
  --training-seed "$TRAIN_SEED" \
  --checkpoint-receipt "$CANDIDATE_OUTPUT/checkpoint_receipt.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --development-holdout-generator-report "$D1_DIR/development_holdout.generator.json" \
  --required-suite dagger1_development \
  --minimum-suites 1 --minimum-episodes-per-suite 1 \
  --minimum-roots-per-suite 30 --max-steps 24 --seed 20260721
```

For batch execution, set `EVALUATION_SCOPE=development_holdout` and the same
study/checkpoint/holdout variables on `submit_dagger_release_eval.sh`. Frozen
evaluation uses `EVALUATION_SCOPE=frozen_suite` (the default) and the manifest's
pinned seed, suite, policy, and 24-step budget. Existing or symbolic-link study
artifact paths are never replaced.

Recovery stress is built only from a clean committed source tree after D0,
natural D1, the development holdout, and observable probe artifacts have been
regenerated and byte-bound:

```bash
python -m psse_env.dagger.build_dagger1_recovery_stress \
  --development-holdout "$D1_DIR/development_holdout.json" \
  --development-holdout-manifest "$D1_DIR/development_holdout.json.manifest.json" \
  --development-holdout-generator-report "$D1_DIR/development_holdout.generator.json" \
  --d0-aggregate-dir "$D0_DIR" \
  --d1-training-scenarios "$D1_DIR/scenarios.json" \
  --d1-training-manifest "$D1_DIR/scenarios.json.manifest.json" \
  --recovery-probes "$D1_DIR/recovery_probes.jsonl" \
  --recovery-probe-manifest "$D1_DIR/recovery_probes.manifest.json" \
  --frozen-suite psse_env/dagger/suites/bc0_eval_suite_v1.json \
  --output "$D1_DIR/recovery_stress.json"
```

The builder derives deterministic intervention states from 20 roots in the
protected 30-root development parent. Those states are absent from the normal
development rollout, and their roots have zero overlap with D0, natural D1,
probe training, and frozen evaluation. It publishes exactly ten episodes in
each of seven strata. Each candidate is executed through the production
environment and independently classified by the observable expert from the
post-intervention policy observation; no expected action is embedded in the
model-visible scenario.

Evaluate it through `submit_dagger_release_eval.sh` with
`EVALUATION_SCOPE=recovery_stress`, `RECOVERY_STRESS_SUITE`, and
`RECOVERY_STRESS_MANIFEST`, plus the same study/checkpoint variables used for
the other scopes. The launcher pins seed `20260723`, 24 steps, all seven suite
names, ten episodes and roots per suite, and writes a
`recovery_stress_evaluation` ingestion report. Recovery stress is not
diagnostic-only and cannot be supplied through the development-holdout path.

Study evaluations are emitted as closed-loop schema v4. Each learner trace row
persists the exact policy-visible observation and tool output, its observation
content hash, a canonical observable-expert action assessment, and narrow
WLS/verification evidence when that tool ran. One evaluator-owned pure
validator is used by both the release gate and study ingestion. It reconstructs
the four-event observation history and last-tool fields from preceding trace
transitions, binds active/candidate identifiers to `state_before`, recomputes
the assessment, and recomputes the narrow certificate from the tool output.
Thus a rehashed observation/assessment or fabricated recovery history cannot
create an opportunity denominator. This supplies exact denominators for all
seven named recovery-action strata and autonomous-exhaustion operator handoffs.
Forbidden-key scanning is case- and separator-insensitive and covers both
observations and policy-visible transitions.

The validator also recomputes every trace state SHA-256 from the exact persisted
canonical lifecycle snapshot, enforces the adjacent state chain, and binds its
first and last active-state identities to the evaluator-owned initial/final
identities in the privileged offline audit. A coordinated trace substitution
therefore cannot become evidence merely by recomputing the trace-local hashes.

The same v4 evidence binds final residual/chi-square acceptance to the final
active state and binds post-commit feasibility to the verification of the exact
candidate that was promoted. Every successful WLS/verification certificate
requires an exact action state ID plus identical, non-null lowercase SHA-256
certificate and trace runtime-state hashes. Physical feasibility is deliberately limited to
the complete MATPOWER observed-snapshot connectivity, voltage, and active
RATE_A certificate. Missing, mismatched, non-observable, incomplete, or
internally inconsistent evidence fails closed. A complete certificate with a
physical violation is counted as a physically unsafe commit. Historical
schema-v3 artifacts may still be ingested for diagnosis, but all of these v4
objectives and safety gates remain explicitly unevaluable and cannot pass.

## Training invocation

The manifest pins the complete material SFT protocol. Both stages use the
reviewed Gemma/processor revision, `AutoProcessor`, `max_length=8192`, QLoRA
NF4 with double quantization and bf16 compute, batch size 1, gradient
accumulation 4, LoRA rank/alpha 16, dropout 0, the seven registered language
projection suffixes, `bias=none`, `task_type=CAUSAL_LM`, `adamw_torch`, the
linear scheduler, and `max_steps=-1`. BC0 is exactly two epochs at `1e-4`;
both Round-1 arms are exactly one epoch at `3e-5`. The manifest also binds the
SHA-256 of `psse_env/requirements-sft.txt`. The study launcher rejects an
environment override that changes any exposed value; the training preflight
reconstructs the complete configuration and compares it to the manifest before
allocating the model.

The 8,192-token value is a reviewed training-resource envelope, not the Gemma
4 architecture limit. The pinned tokenizer advertises 262,144 tokens and the
current D0 aggregate peaks at 6,926. Prompt truncation remains disabled: a
future row above 8,192 must trigger an explicit protocol and memory review, not
silent data loss or an assertion that the row is semantically unqualified.

The SFT CLI accepts one explicit unsigned 32-bit seed:

```bash
python -m psse_env.sft train \
  --seed 3407 \
  --revision "$MODEL_REVISION" \
  --train "$TRAIN_FILE" \
  --validation "$VALIDATION_FILE" \
  ...
```

There is no CLI seed default: omitting `--seed` is an argument error. Likewise,
`STAGE=round0` and `STAGE=round1` reject an absent `TRAIN_SEED` before loading
the training environment or model.

Before base-model construction, training seeds Python `random`, NumPy, Torch
CPU, and all Torch CUDA generators. Cold BC0 construction repeats the same
four-engine reset immediately before attaching the new LoRA adapter. Warm
Round-1 runs do not construct a cold adapter, so that second reset is recorded
as the canonical not-applicable state. The exact RNG contract and both phase
records are included in the checkpoint receipt.

The staged Slurm wrapper binds the same value for Round 0 and Round 1:

```bash
sbatch \
  --export="ALL,STAGE=round0,TRAIN_SEED=3407,REVIEWED_SOURCE_COMMIT=$FREEZE_COMMIT" \
  submit_dagger_sft_round0.sh
```

Round 1 additionally requires the canonical BC0 receipt beside the initial
adapter:

```bash
export INITIAL_ADAPTER_PATH=/absolute/bc0-seed3407/lora
export INITIAL_ADAPTER_REVISION=<bc0-adapter-tree-sha256>
export PARENT_CHECKPOINT_RECEIPT=/absolute/bc0-seed3407/checkpoint_receipt.json
```

That receipt must validate as BC0 under the same study manifest, reviewed
source commit, and training seed. Its recorded adapter path and tree SHA-256
must exactly match the warm-start arguments. Validation finishes before model
allocation, and the new Round-1 receipt binds the authenticated
`parent_checkpoint_receipt_id`. BC0 records the canonical null parent ID.
The canonical D1 aggregate independently retains the adapter-tree identity
used for learner-in-the-loop collection. That collector identity authenticates
the shared dataset; it does not override the same-seed BC0 parent identity for
replicated Round-1 training.

Use a distinct output directory and artifact identity for every variant/seed.
When W&B is enabled, the default run ID/name and tags include `TRAIN_SEED`, so
replicates do not collide. Custom W&B identifiers remain the operator's
responsibility and must not merge different seeds.

After the model and processor are saved and the adapter base reference is
normalized, every training run inspects the complete final `lora/` tree and
writes a fully flushed temporary receipt and atomically hard-links the
write-once `checkpoint_receipt.json` beside it. The receipt is outside `lora/`,
so it cannot participate in or self-reference the canonical adapter-tree
digest. The receipt binds the
seed, variant, manifest digest, clean reviewed source commit, exact training-
view provenance and split hashes, exact training protocol and dependency lock,
RNG attestation, parent revision and receipt ID, base model, and final adapter
tree SHA-256. Natural/full receipts additionally bind the complete Round-1
production-D1 zero-quarantine report to the same generation-provenance ID;
BC0 carries the one canonical not-applicable/null object. The receipt also
binds its exact schema version and base-snapshot-attestation SHA-256 and embeds
the actual runtime accelerator attestation—GPU
name, approved `h100`/`h200`/`rtx6000` class, memory, compute capability, CUDA
version, and bf16 support. Validation recomputes the class from the name and
approved memory floor, so a self-rehashed receipt cannot relabel another or
undersized GPU—while still allowing any approved class for portable study
runs. Before training, any existing or dangling-symlink checkpoint-owned output
(`lora/`, either model attestation, the receipt, or a known staging path) makes
the command fail instead of reusing an interrupted run. Invalid source/view
binding, source-set contamination, an invalid adapter tree, receipt
replacement, or a tree change during publication also fails closed.

## Local contract check

```bash
python - <<'PY'
from psse_env.dagger.study_manifest import load_study_manifest

manifest = load_study_manifest()
print(manifest["manifest_sha256"])
print(manifest["validation"])
PY
```

Any JSON byte change, missing/extra variant, duplicate or invalid seed, source
relaxation, model drift, suite/policy drift, or incomplete artifact binding
fails closed.
