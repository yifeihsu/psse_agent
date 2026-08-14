# Live Gemma 4 gate results — 2026-07-15

These checks were rerun after the final pilot regeneration and use the pinned
Hugging Face cache. They are separate from the offline fake-processor unit
suite. Machine-readable reports are under `psse_env/examples/sft_pilot/`.

## Final grouped pilot

`generate_sft_pilot` produced 90 deterministic, production-tagged rows from 15
root scenarios. The split was performed before chat export:

| Split | Rows | Root groups |
| --- | ---: | ---: |
| train | 78 | 13 |
| validation | 6 | 1 |
| test | 6 | 1 |

The native chat audit passed 90/90 rows. Teacher realizability found 38 unique
canonical observations, zero conflicting observations, and conflict rate 0.0.
The target-aware class audit passed with 60 `clean_successful`, 15
`accepted_final_commit`, and 15 `terminal_decision` rows.

Correction prompts retain bounded observable family findings and the exact
provider-derived `supported_corrections`; pilot scenarios contain no oracle
action hints. Production collection rejects a correction target that does not
match that visible evidence, and ambiguous multi-family context routing uses a
deterministic observable tie-break.

This is a tokenizer and training-stack pilot, not a recovery evaluation. It has
no rejected-candidate, partial-commit, invalid-precondition, rollback, or loop
rows; validation and test each contain only one root group.

## Exact 31B processor/template gate

Command run from the archive root:

```bash
python -m psse_env.sft gate \
  --model unsloth/gemma-4-31B-it \
  --revision 8a796db4df380b178065ed910849477ff0e99c87 \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl \
  --test psse_env/examples/sft_pilot/pilot.test.jsonl \
  --max-length 4096
```

`AutoProcessor` / `Gemma4Processor` passed all 90 rows. All assistant tool calls
parsed back to the expected dictionary-valued canonical call.

| Split | p50 | p95 | p99 | max | Target truncation | Zero supervision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2,070 | 2,440 | 2,440 | 2,440 | 0 | 0 |
| validation | 2,048 | 2,397 | 2,397 | 2,397 | 0 | 0 |
| test | 2,099 | 2,440 | 2,440 | 2,440 | 0 | 0 |

Prompt truncation was also zero. The pinned 31B template requires explicit
empty-thought alignment; the gate injected it on all 90 rows before establishing
the exact completion boundary. No substitute chat template was used.

The reproducible all-tool probe additionally targeted every one of the 13 macro
actions while retaining the complete 13-schema tool set on every row:

```bash
python -m psse_env.examples.run_exact_tool_probe
```

| Processor | Round trips | Length p50 / p95 / p99 / max | Gate |
| --- | ---: | ---: | ---: |
| E2B `f0c5915f...` | 13/13 | 1,249 / 1,264 / 1,264 / 1,264 | pass |
| 31B `8a796db4...` | 13/13 | 1,253 / 1,268 / 1,268 / 1,268 | pass |

Both probes had zero prompt truncation, target truncation, and zero-supervision
rows. The adversarial episode ID was the single character `e`, confirming that
identifier aliasing does not corrupt tool names.

## Final local E2B QLoRA gates

Hardware: NVIDIA RTX 4080. Model and processor both used
`unsloth/gemma-4-E2B-it@f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae`,
cache-only. The model used NF4 double quantization with bfloat16 compute,
`prepare_model_for_kbit_training()`, and LoRA limited to exact language-tower
projection paths.

- One-batch forward/backward: pass; loss `14.1112756729`; finite gradients;
  trainable parameter changed.
- 20-step tiny overfit at learning rate `0.001`: pass; loss
  `14.1112756729 -> 0.0005142066`; finite gradients; trainable parameter changed.
- Greedy generated-output round trip: pass; parsed target tool `run_wls` with
  the expected dictionary arguments.

The optimizer commands used the current entrypoint and stopped before TRL. The
trained first row (`dagger_iter0_pilot_root_000_step0`) is byte-identical after
the later observable-context export fix (row SHA-256 `bc685c77...`). The exact
E2B processor gate was then rerun over all 90 final rows: 90/90 round trips,
maximum 2,436 tokens, and zero prompt/target truncation or zero-supervision rows.

## Launch boundary

Status is **GO for the corrected 90-row export/tokenizer pilot and local E2B
QLoRA stack**. Status remains **NO-GO for a full 31B SFT run** because no 31B
model forward/backward or tiny-overfit was performed and the pilot does not
exercise recovery classes. Run those exact-checkpoint gates on suitable HPC
hardware, generate a recovery-balanced aggregate, and pass a short held-out
recovery evaluation before full training.
