# Gemma 4 tool-call SFT

This package turns DAgger chat rows into Gemma 4 LoRA training runs. It renders
every row with the processor's own `apply_chat_template` (there is no
hand-written template fallback), masks every prompt token with `-100`, checks
that the assistant target survives truncation and parses back to the canonical
tool call, and trains with TRL.

Every input row must contain:

- a non-empty row-level `tools` list of JSON function schemas;
- `messages` whose final message is the assistant target;
- dictionary-valued `function.arguments` for every assistant tool call;
- a non-empty `root_scenario_id`.

## Training

The pipeline stages (`research/hpc/full_pipeline_20260907/stage_bc0.sbatch`
and `stage_train.sbatch`) call the research trainer:

```bash
python -m psse_env.sft research-train \
  --model-choice 12b \
  --train <train.jsonl> \
  --validation <validation.jsonl> \
  --output-dir <run-dir> \
  --learning-rate 1e-4 --epochs 1 \
  --save-steps 64 --eval-steps 64 --select-best-eval-loss
```

`--model-choice` selects a pinned preset from `psse_env/research_models.py`
(`12b` for reportable runs, `e4b` for fast smoke runs). `--initial-adapter`
continues a previous adapter (the DAgger rounds), `--smoke-steps` runs an
optimizer smoke step before training, and `--resume-from-checkpoint` resumes
an interrupted run; `python -m psse_env.sft.research_checkpoint --output-dir
<run-dir>` prints the newest complete checkpoint. The finished adapter is
written to `<run-dir>/lora`.

| Module | Role |
|---|---|
| `research_cli.py` | `research-train` entry point |
| `research_rows.py` | Normalizes rows to the current tool registry |
| `gates.py` | Row, schema, rendering and assistant-mask checks |
| `collator.py` | Padding collator that preserves `-100` labels |
| `training.py` | LoRA and TRL configuration |
| `smoke.py` | One-batch smoke and single tool-call generation |
| `gemma_text.py` | Prompt rendering, stop tokens and decoding shared with evaluation |
| `research_checkpoint.py` | Finds the newest complete checkpoint to resume from |
