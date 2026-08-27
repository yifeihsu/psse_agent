# Gemma-4-12B BC0 audited repair and checkpoint handoff

Date: 2026-08-25  
Code commit: `27b0a3bee5593b39a738063ccf2cdc07b28ef362`  
Slurm job: `16323924` (`COMPLETED`, exit `0:0`, elapsed `01:03:48`)  
GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition  
Training performed: none; all adapter weights were replayed unchanged.

## Repair audit

The repair adds the missing canonical `line_index1` to internal `line_index`
translation for topology corrections and injects
`psse_env.dagger.release_factories:deterministic_case_loader` into research BC0
evaluation.

| Audit item | Historical evaluator | Repaired replay |
|---|---:|---:|
| D0 episodes with `case_loader_required_for_physical_comparison` | 13/21 | 0/21 |
| D0 episodes receiving strict release audit | 8/21 | 21/21 |
| D1 episodes with `case_loader_required_for_physical_comparison` | 9/15 | 0/15 |
| D1 episodes receiving strict release audit | 6/15 | 15/15 |
| Historical D0 topology actions | 0/8 succeeded; all retained `line_index1` | n/a |
| Repaired D0 topology actions | n/a | 5/5 succeeded; all reached the controller as `line_index` |

Topology action counts differ because the repaired controller changes the later
closed-loop observations and therefore the generated trajectory. The important
execution audit is that no repaired topology action retained `line_index1` at
the internal controller boundary and none failed there.

## Same published adapter replay

The published adapter is byte-identical in weights to checkpoint 128
(`adapter_model.safetensors` SHA-256
`605cb5aa29432d84edabba8e1529215322b708f2b90f003d54c9eec0f43a8870`).

| Phase | Strict resolved physical success | Audited post-correction handoff | Audited completion union | Schema valid | Expert tool agreement | Evaluator errors |
|---|---:|---:|---:|---:|---:|---:|
| Repaired D0, 21 roots | 4/21 (19.0%) | 8/21 (38.1%) | 12/21 (57.1%) | 99.5% | 68.2% | 0 |
| Repaired D1, 15 protected roots | 0/15 (0.0%) | 7/15 (46.7%) | 7/15 (46.7%) | 100.0% | 69.0% | 0 |

The historical v1 reports recorded D0 strict resolution as 4/21 and D1 strict
resolution as 0/15. They could not validly audit all handoffs because the case
loader was absent. Their zero handoff fields must therefore not be interpreted
as audited negative outcomes. The repaired v2 contract reports strict
resolution and safety-clean operator handoff separately.

## Existing checkpoint comparison on the same frozen D1 suite

All checkpoints were evaluated sequentially with seed `20260720`, maximum 24
closed-loop steps, the same 15 scenarios, and the same 15 physical roots.

| Rank | Checkpoint | SFT eval loss | Strict resolved | Audited handoff | Schema valid | Expert tool agreement | Invalid actions | False commits | Loop episodes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 192 | 0.024009 | 0/15 | 9/15 (60.0%) | 99.4% | 71.4% | 11 | 6 | 0 |
| 2 | 320 | 0.026022 | 0/15 | 8/15 (53.3%) | 100.0% | 70.2% | 20 | 5 | 5 |
| 3 | 256 | 0.031865 | 0/15 | 8/15 (53.3%) | 99.3% | 68.8% | 26 | 4 | 6 |
| 4 | 128 (published) | 0.022957 | 0/15 | 7/15 (46.7%) | 100.0% | 69.0% | 24 | 7 | 3 |
| 5 | 64 | 0.037447 | 0/15 | 4/15 (26.7%) | 98.1% | 59.1% | 43 | 11 | 2 |

Checkpoint 192 is the best saved checkpoint for audited research handoff on
this small protected suite, despite checkpoint 128 having the lowest SFT
validation loss. No checkpoint achieved strict resolved physical success, so
checkpoint 192 is not a strict-success promotion. The published adapter was not
replaced or modified.

## Verification and limits

- Local test suite before deployment: 282 passed, 1 skipped, 64 subtests passed.
- Remote CPU smoke: canonical topology conversion passed; deterministic case14
  loading produced 14 buses and 20 branches.
- Slurm stderr contains no traceback, runtime exception, OOM, or missing-file
  error. All seven copied key artifacts match their remote SHA-256 hashes.
- GPU telemetry: 375 ten-second samples; 55.7% mean GPU utilization, 64.1% mean
  utilization while active, 20,863 MiB peak memory, and 495.66 W peak power.
- This is a single-seed, 15-root research comparison. It supports a checkpoint
  selection hypothesis, not a generalization or production-readiness claim.

