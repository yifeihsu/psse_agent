# Gemma 4 SFT go/no-go gate

This package is the production boundary between DAgger chat rows and Gemma 4
LoRA/TRL training. It does not import the root training script and it does not
contain a fallback chat template.

Pinned processor and local E2B QLoRA smoke results are recorded in
[`LIVE_GATE_RESULTS.md`](LIVE_GATE_RESULTS.md). The report distinguishes the
31B processor-only result from the E2B model optimizer checks.

Every input row must contain:

- a non-empty row-level `tools` list of JSON function schemas;
- `messages` whose final message is the assistant target;
- dictionary-valued `function.arguments` for every assistant tool call;
- a non-empty `root_scenario_id` used for grouped split validation.

Run the mandatory live gate with the same model id and pinned revision that the
training job will use:

```bash
python -m psse_env.sft gate \
  --model unsloth/gemma-4-31B-it \
  --revision 8a796db4df380b178065ed910849477ff0e99c87 \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl \
  --test psse_env/examples/sft_pilot/pilot.test.jsonl
```

The default is cache-only. Add `--allow-download` when the HPC node is expected
to fetch from Hugging Face. Missing Transformers, missing model files, an
unpinned revision, an unsupported template, or any rendering error returns exit
code 2 and a `passed: false` result. Offline fake-processor tests are useful but
never satisfy this live gate.

The grouped gate also rejects any row that is not tagged `dataset_mode=production`
at both row and metadata level. It renders `tools` through the exact processor's
`apply_chat_template()`, proves that the generation prompt is an exact token
prefix, masks every prompt token with `-100`, checks that the assistant target
survives truncation, parses rendered tool calls back to a dictionary-valued
canonical call, reports p50/p95/p99/max lengths, and rejects zero-supervision
rows. Prompt truncation is also a failure unless explicitly approved.

Run either optimizer gate without starting TRL training:

```bash
python -m psse_env.sft smoke ... --mode one-batch
python -m psse_env.sft smoke ... --mode tiny-overfit --tiny-overfit-steps 20
```

The smoke command defaults to 4-bit QLoRA and stops immediately after the
requested gate. For a 32-128 row grouped pilot, gate and launch LoRA/TRL training
with a required forward/backward optimizer smoke step:

```bash
python -m psse_env.sft train \
  --model unsloth/gemma-4-31B-it \
  --revision <pinned-hf-commit> \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl \
  --output-dir outputs/dagger_gemma4_pilot \
  --smoke-steps 1 \
  --load-in-4bit
```

Use `--smoke-steps 20` for a tiny-overfit gate; more than one step additionally
requires the final loss to be lower than the initial loss. The entrypoint uses
standard language-model LoRA targets (`q/k/v/o`, gate, up, and down projections),
a pretokenized custom collator, TRL `SFTConfig`, and `SFTTrainer`.

The bundled result is **GO only for the corrected pilot and local E2B stack**.
No 31B model forward/backward was run. Do not launch full 31B SFT until that
exact checkpoint passes its one-batch and tiny-overfit gates on suitable HPC
hardware and a recovery-balanced held-out evaluation passes. The bundled pilot
contains successful measurement/parameter/topology paths but no rollback,
partial-commit, invalid-precondition, or loop classes.

## NYU HPC staged pilot

The root `submit_dagger_sft_pilot.sh` launcher targets this package. The older
`submit_sft_gemma4.sh` invokes a different trainer and dataset and must not be
used for this pilot. Create a fresh environment from `requirements-sft.txt`,
then submit the following stages on one high-memory A100, H100, or H200:

1. `STAGE=gate` checks all 90 rows with the pinned processor.
2. `STAGE=one-batch` runs the exact 31B forward/backward QLoRA gate.
3. `STAGE=tiny-overfit` requires a loss decrease over 20 steps and an exact
   greedy tool-call round trip.
4. After reviewing those logs, `STAGE=pilot` trains only the bundled 90-row
   pilot and writes the final adapter to `<output-dir>/lora`.

The launcher deliberately refuses `STAGE=full` and `STAGE=production`. It is a
single-process, single-GPU pilot launcher with no checkpoint-resume callback;
it is not the future full-data production launcher.
