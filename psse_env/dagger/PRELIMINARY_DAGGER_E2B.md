# Preliminary DAgger E2B run

`submit_preliminary_dagger_e2b.sh` is a fast diagnostic path for obtaining an
early DAgger signal. It does not replace the pinned 31B multiseed study and its
outputs are never release eligible.

The job accepts only
`preliminary_dagger_dataset_receipt_v1` with
`release_eligible: false`. Before loading a model, it rehashes all seven JSONL
files, re-extracts physical roots, and requires BC0/D1 plus D1
train/validation/test roots to be disjoint. D1 validation is used for Trainer
evaluation. D1 test and the combined D1 evaluation file are hash-bound and
reserved for a separate post-training closed-loop evaluation; neither is
optimizer-visible.

The dataset builder itself also fails closed unless its module and command-line
wrapper are tracked and byte-identical to the current Git `HEAD`. The dataset
receipt records their commit, Git blob IDs, SHA-256 hashes, and sizes; the
launcher validator requires that source attestation before accepting any data.

The fixed model is `unsloth/gemma-4-E2B-it` at Hugging Face revision
`f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae`. The first stage runs 64 bounded
BC0 steps. The second stage loads that BC0 LoRA and runs 96 bounded steps on the
mixed BC0+D1 view. Both stages use an 8,192-token context and pass
`--fail-on-prompt-truncation`, `--preserve-system-text`,
`--no-phase-gated-prompt`, and `--sanity-check-samples 0`, because recovery
targets are not universally first-turn WLS targets. Before warm-start loading,
the job copies BC0 into a bound initialization adapter whose
`base_model_name_or_path` is the local snapshot directory for that exact E2B
commit. This closes the legacy trainer's behavior of not forwarding
`--model-revision` when it loads a local adapter.

On an offline compute node, `MODEL_NAME` may point to the absolute local Hub
snapshot directory only when its basename is that exact pinned revision and it
contains `config.json`. The launcher rejects every other override, and stage
receipt validation treats the canonical Hub ID and that exact local snapshot
as the same pinned model identity.

Both stages evaluate on the identical root-held-out
`preliminary.d1_validation.jsonl`. Their recorded `eval_loss` values therefore
provide a fast before/after DAgger signal on comparable bytes. The comparison
is still confounded by continued training and is neither a closed-loop result
nor release evidence. `preliminary.d1_test.jsonl` remains completely untouched
for a later evaluation.

Before accepting any Trainer state, each output directory gets a write-once
`preliminary_stage_plan.json`. It binds the dataset receipt and exact
train/validation hashes, model revision, seed, row caps, sequence length,
batch/accumulation profile, learning rate, LoRA arguments, save/evaluation
cadence, and the warm-start adapter where applicable. A checkpoint is resumable
only when its numeric step, `trainer_state.json`, and existing `run_config.json`
match that plan. A stale checkpoint without a matching plan is rejected. Stage
completion additionally requires a finite `eval_loss` at the planned final
step, over every D1 validation row (the default cap of 128 covers the current
63-row validation file). It then greedily decodes a deterministic 32-row,
root-spread validation sample. Both stages require schema-valid tool calls on
at least 99% of rows, controller-bound state references on at least 98%, and no
64-token-limit hits. DAgger also requires at least 50% target-tool agreement;
BC0 records that metric without gating it so a weak baseline remains a valid
comparator. The generation report is bound into the stage receipt. Step
overrides must remain divisible by both the save and evaluation cadence so the
final-step evidence exists.

The plan also attests the clean committed bytes of the preliminary launcher,
receipt/hardware/adapter helpers, legacy Trainer, and its direct local adapter,
tool-schema, and release-hardware dependencies. Unrelated worktree changes may
coexist, but any changed or untracked file in that execution surface blocks
plan publication. For a DAgger stage, the plan inspects the exact
optimizer-visible mixed-data prefix and refuses to label the run DAgger unless
that prefix contains at least one D1 row/root; the normal 512-row cap contains
substantially more. Trainer completion must also show that preprocessing kept
every bound prefix row.

Build the dedicated dataset first. The default builder selects 525 D1 training
rows and reserves 30 whole held-out D1 roots, split 15/15 between validation
and test:

```bash
python scripts/build_preliminary_dagger_dataset.py \
  --failure-evidence /path/to/strict_failure_evidence.json \
  --candidate-rows /path/to/diagnostic.candidate_recovery_rows.jsonl \
  --d0-generation-provenance /path/to/round0/aggregate.generation_provenance.json \
  --d0-train /path/to/round0/aggregate.train_view.jsonl \
  --d0-validation /path/to/round0/aggregate.validation.jsonl \
  --output-dir /scratch/yx3882/preliminary_dagger
```

Submit on any approved high-memory route:

```bash
sbatch \
  --export=ALL,DATASET_RECEIPT=/scratch/yx3882/preliminary_dagger/preliminary.dataset_receipt.json \
  submit_preliminary_dagger_e2b.sh
```

For NYU's RTX Pro 6000 route, use its `rtx6000` Slurm feature spelling:

```bash
sbatch --constraint=rtx6000 --cpus-per-task=4 --mem=96G \
  --export=ALL,EXPECTED_ACCELERATOR_CLASS=rtx6000,DATASET_RECEIPT=/scratch/yx3882/preliminary_dagger/preliminary.dataset_receipt.json \
  submit_preliminary_dagger_e2b.sh
```

The runtime attestation accepts one H100, H200, actual RTX Pro 6000 with at
least 90,000 MiB, or exact NVIDIA L40S with at least 45,000 MiB. It rejects the
older 48-GB RTX 6000 Ada card. L40S uses batch size 2 and gradient accumulation
8 by default; the other routes use 8 and 2, so every default keeps an effective
batch of 16. This L40S exception exists only in the preliminary hardware gate;
the release gate is unchanged. Submit the currently available L40S route with:

```bash
sbatch --constraint=l40s \
  --export=ALL,EXPECTED_ACCELERATOR_CLASS=l40s,DATASET_RECEIPT=/scratch/yx3882/preliminary_dagger/preliminary.dataset_receipt.json \
  submit_preliminary_dagger_e2b.sh
```

Set
`ALLOW_DOWNLOAD=1` only for the first pinned E2B snapshot download; later jobs
default to cache-only mode.

Slurm preemption saves a Trainer checkpoint and requeues the job. Completed
stages have write-once `preliminary_stage_receipt.json` files bound to the
dataset receipt, immutable stage plan, Trainer run configuration, split bytes,
pinned model, hardware attestation, final-step evaluation, and LoRA tree. A replacement job
skips a valid completed stage, resumes an incomplete stage only from a concrete
Trainer checkpoint, and refuses an ambiguous unreceipted LoRA tree.

Useful bounded overrides are `BC0_MAX_STEPS` (maximum 64),
`DAGGER_MAX_STEPS` (maximum 128), and `PIPELINE_STAGE=bc0|dagger|all`. The
sequence length is pinned to 8,192 because the measured canonical prompts do
not safely fit the old 4,096-token setting. The stage
receipts expose the latest recorded training/evaluation losses for quick
inspection, but those losses are diagnostic—not a preliminary closed-loop
success claim. Use the held-out D1 files in a separate evaluation for that
result.
