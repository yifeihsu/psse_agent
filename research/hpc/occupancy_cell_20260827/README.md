# Occupancy screening cell (2026-08-27)

This is a research-only screening cell. It validates the DAgger pipeline; it is
not a release path and it does not run DAgger-2.

| Arm | Model / corpus | Optimizer updates | Interpretation |
| --- | --- | ---: | --- |
| A | E2B learner-full | 666 | fixed-update learner-occupancy screen |
| B | 12B learner-full | 662 | fixed-update learner-occupancy screen |
| C | E2B legacy mixture-full | 1811 | legacy row-pass/coverage match |
| D | 12B legacy selective | 247 | legacy row-pass/coverage match |

The E2B control actually completed 666 optimizer updates, despite the earlier
matrix describing the shared budget as 662, so A uses 666. B retains the
observed 12B value of 662. The 1811/247 values are the exact integer budgets
from the prepared-row counts and Trainer's `ceil(rows / 4)` update cadence;
they supersede the approximate 1780/246 planning values.

The 1811/247 budgets reproduce the historical row-pass comparison. They are
not target-token-matched budgets. A/B use the new audited-universe split;
C/D retain their legacy split receipts. Therefore this four-job cell is a
provisional crossover screen, not a clean two-model by three-occupancy
factorial.

## Fail-closed launch

1. Deploy one committed source snapshot and put all inputs at immutable paths.
2. Copy `config.example.json`, replace every placeholder, and compute the source
   digest with `python build.py tree-digest --root /absolute/source/root`.
   Compute each offline model snapshot digest with `python build.py
   snapshot-digest --hf-home /absolute/hf/home --lane e2b` (and `--lane 12b`)
   and bind the reported hashes in the configuration.
3. Recheck every configured SHA-256 and the 12B selective inclusion receipt.
   Arm C has no historical E2B aggregate report, so the CPU build proves its
   inclusion directly by reconstructing all audited D1 chat rows, checking the
   D0 prefixes, comparing the full row multiset, and checking root isolation.
4. Recheck current Torch feature names/capacity, then submit:

```bash
export CELL_CONFIG=/absolute/path/cell.config.json
export CELL_PYTHON=/scratch/yx3882/envs/dagger12b_overlay/bin/python
export CELL_GPU_CONSTRAINT='e2b=RECHECK_FEATURE,12b=RECHECK_FEATURE'
export CELL_GPU_FAMILY='e2b=RECHECK_FAMILY_SET,12b=RECHECK_FAMILY_SET'
bash research/hpc/occupancy_cell_20260827/submit.sh
```

Current routing blocker: the historical E2B arms ran on A100 and the historical
12B arms ran on H100, but exact A100/H100 requests are not currently accepted
by Torch; the authorized union also exposes H200 and `rtx6000` candidates.
`submit.sh` deliberately has no routing default. Both
the scheduler feature and expected physical family set must be explicit. Family
sets may contain `A100`, `H100`, `H200`, or `RTX6000`, joined by `|`; RTX 6000
matches both the legacy spelling and Torch's observed `NVIDIA RTX PRO 6000
Blackwell Server Edition` name. Every GPU job requires exactly
one CUDA-visible device, derives its family from the Torch device-0 properties,
and records its exact name, memory, compute capability, CUDA binding, node, and
inventory before training. No script specifies a partition.

A scheduler union is not an actual hardware match: A/C or B/D can land on
different devices. Compare a pair as hardware-matched only when the two
training gates report the same exact allocated-device name, memory, and compute
capability. Otherwise retain the results as pipeline screens and label the
hardware mismatch as a confound.

The explicitly authorized scheduler-union form is:

```bash
export CELL_GPU_CONSTRAINT='e2b=a100|h100|h200|rtx6000,12b=a100|h100|h200|rtx6000'
export CELL_GPU_FAMILY='e2b=A100|H100|H200|RTX6000,12b=A100|H100|H200|RTX6000'
```

The submit launcher refuses to run unless its resolved directory is inside the
configured, tree-hashed `source_root`. Because Slurm executes a spool copy of
each batch script, every job byte-compares that copy with the corresponding
script under the hashed source tree before using any gate code. Only the
explicitly required variables and `PATH` are exported to jobs. If an `sbatch`
or receipt update fails during
submission, the launcher requests cancellation of every job ID created by that
attempt and records those IDs in `submission.json`; inspect that receipt and
`squeue`/`sacct`, then use a fresh cell root for a new attempt.

The CPU build job constructs A/B aggregates on the same audited-root split and
runs Linux tests plus processor-only exposure checks for all four arms. The
environment gate pins Python 3.12.12, torch 2.10.0, transformers 5.15.1, TRL
1.10.0, PEFT 0.20.0, bitsandbytes 0.49.2, and datasets 5.0.1. It deliberately
does not use `pip check`: this historical overlay has known package-metadata
conflicts even though these are the exact experiment versions. Each GPU job starts in a
fresh `job-ID/attempt-rN` directory, trains from the base model, emits a full
schema-v2 65-episode evaluation, and passes exact hash/model/update gates. A
dependent CPU job then deterministically replays that evaluation and requires
65 physically assessable episodes, zero replay mismatches, and zero
unclassified outcomes. Preemption/requeue creates a new restart attempt rather
than reusing a partial adapter.

The configuration pins `HF_HOME` to the historical offline cache. The CPU build
strictly resolves every entry in both exact-revision snapshots, rejects targets
outside that cache, hashes all exposed bytes, and compares the two deterministic
manifest digests with the immutable configuration. Each GPU job repeats this
content check for its own model lane before loading weights. The later physical
audit does not depend on continued model-cache availability.

All job IDs, constraints, input/source hashes, exact sampled training exposure,
evaluation hashes, and physical summaries are retained below `cell_root`.
Torch currently omits `SLURM_JOB_CONSTRAINTS` inside the batch environment, so
each GPU arm reads the job-level `Constraints` field back from `sacct`, requires
an exact match with the submitted feature expression, and includes that value
and its scheduler source in the hashed allocated-device receipt before training.
`source_commit` is an informational deployment label because a `git archive`
has no `.git` directory; the validated `source_tree_sha256` is the authoritative
source binding.
