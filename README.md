# PS-LLM-Agent

PS-LLM-Agent is a power-system diagnostic agent project built around MATPOWER-backed tool use and Gemma 4 supervised fine-tuning. The active root-level pipeline in this repo is:

`preprocess.py` -> `submit_sft_gemma4.sh` / `gpt_oss_power_sft_revised_v3.py` -> `submit_eval_v3.sh` / `eval_sft_agent_gemma_v4.py`

The current training and evaluation flow targets Gemma 4 tool-calling traces stored as JSONL conversations in `artifacts/traces/out_traces_balanced/`.

## Active Workflow

1. Generate or collect raw diagnostic traces.
2. Normalize and split them with `preprocess.py`.
3. Fine-tune Gemma 4 with `submit_sft_gemma4.sh` or `gpt_oss_power_sft_revised_v3.py`.
4. Evaluate the LoRA adapter with `submit_eval_v3.sh` or `eval_sft_agent_gemma_v4.py`.

## Repository Layout

- `Transmission/`: MATLAB and Python data-generation code for SE, topology, parameter-error, and HSE scenarios.
- `Harmonics/`: Harmonics-specific generation and verification utilities.
- `mcp_server/`: MATPOWER-backed tool implementations used by evaluation and external MCP-style agent runs.
- `data/`: Source JSONL datasets such as `data/sft_with_tools.jsonl`.
- `artifacts/measurements/`: Generated raw measurement datasets.
- `artifacts/traces/`: Generated SFT/tool-use trace datasets and inspection subsets.
- `artifacts/traces/out_traces_balanced/`: Preprocessed train/valid/test traces used for SFT.
- `artifacts/eval_archives/`: Evaluation result archives and packaged review outputs.
- `artifacts/logs/`: Slurm and local run logs.
- `outputs/`: Fine-tuning checkpoints, LoRA adapters, and evaluation outputs.
- `docs/`: Notes, slides, and supporting documentation.
- `docs/hif_multiscan_estimation.md`: Multi-scan HIF window generation, estimator semantics, observability diagnostics, and validation gates.
- `docs/hif_multiscan_benchmark_20260713.md`: Initial paired snapshot-count benchmark.
- `docs/hif_multiscan_revision_20260714.md`: Transformer correction, strict 17-event QA, controlled A-E benchmark, and production go/no-go decision.

## Active Root Scripts

- `preprocess.py`: Normalizes traces, optionally deduplicates them, audits token length, and writes balanced train/valid/test splits.
- `gpt_oss_power_sft_revised_v3.py`: Canonical Gemma 4 SFT entrypoint.
- `submit_sft_gemma4.sh`: Slurm wrapper for training with auto-tuned defaults by GPU class.
- `eval_sft_agent_gemma_v4.py`: Main Gemma 4 evaluator.
- `submit_eval_v3.sh`: Slurm evaluation wrapper with resume support and current runtime defaults.
- `make_stratified_smoke.py`: Builds a small balanced evaluation subset from the test split when you want a faster check.
- `eval_sft_agent_hardened.py`: Shared hardened evaluation/runtime logic reused by the Gemma evaluator.
- `trace_protocol.py`: Shared tool schemas and trace-format helpers.
- `interactive_agent_eval.py`: Manual interactive runner for probing an adapter with the MATPOWER tools.
- `export_hf_prompt_completion.py`: Exports chat traces into flat prompt/completion JSONL files for other HF-style workflows.
- `setup_unsloth_env.sh`: One-time environment bootstrap for the cluster workflow used by the submit scripts.

## Requirements

- Python 3.11 is the target environment used by `setup_unsloth_env.sh`.
- MATLAB with MATPOWER available to the Python-side tool wrappers in `mcp_server/`.
- A CUDA GPU for training or evaluation.
- Slurm if you want to use the provided `submit_*.sh` wrappers.

The submit scripts are currently configured for the NYU Greene-style environment:

- repo checkout at `/scratch/yx3882/psse_agent`
- conda env at `/scratch/yx3882/.conda/envs/unsloth_sft`
- module bootstrap via `anaconda3/2025.06`

If you run elsewhere, either edit the submit scripts or call the Python entrypoints directly.

## Environment Setup

One-time cluster setup:

```bash
bash setup_unsloth_env.sh
```

The script creates `/scratch/yx3882/.conda/envs/unsloth_sft` and installs the training stack used by the current Gemma 4 pipeline:

- `torch`
- `transformers`
- `datasets`
- `trl`
- `peft`
- `bitsandbytes`
- `unsloth`
- `wandb`

## Data Preparation

The active preprocessing path starts from `data/sft_with_tools.jsonl` and writes balanced splits to:

- `artifacts/traces/out_traces_balanced/sft_traces.train.jsonl`
- `artifacts/traces/out_traces_balanced/sft_traces.valid.jsonl`
- `artifacts/traces/out_traces_balanced/sft_traces.test.jsonl`

Basic usage:

```bash
python preprocess.py \
  --input data/sft_with_tools.jsonl \
  --exact-balanced \
  --dedupe-by user_snapshot
```

Useful notes:

- `--exact-balanced` selects exactly 500 samples per error family and splits them 400/50/50 into train/valid/test.
- `--dedupe-by user_snapshot` is a reasonable default when you want to avoid near-duplicate operating points across splits.
- The preprocessing report is written to `artifacts/traces/out_traces_balanced/preprocess_report.json` by default.

## Training

The active trainer is `gpt_oss_power_sft_revised_v3.py`, and the recommended cluster wrapper is `submit_sft_gemma4.sh`.

Important:

- Training requires a pinned `MODEL_REVISION` unless you explicitly opt out with `ALLOW_UNPINNED_MODEL_REVISION=1`.
- The default model is `unsloth/gemma-4-31B-it`.
- The default output directory is `outputs/gemma4_power_agent`.

Recommended Slurm launch:

```bash
export MODEL_REVISION=d722512f8f1e4ef6629c1b24d16d65295c8c945e
sbatch --constraint='a100|h100|h200' submit_sft_gemma4.sh
```

Common overrides:

```bash
export MODEL_REVISION=d722512f8f1e4ef6629c1b24d16d65295c8c945e
export OUTPUT_DIR=/scratch/yx3882/psse_agent/outputs/gemma4_power_agent_exp1
export MAX_SEQ_LENGTH=6144
export PER_DEVICE_TRAIN_BATCH_SIZE=2
export GRADIENT_ACCUMULATION_STEPS=8
export WANDB_PROJECT=psse-agent-sft
sbatch --gres=gpu:rtx_pro_6000:1 submit_sft_gemma4.sh
```

Direct Python launch is also possible if your local environment already matches the needed dependencies:

```bash
python gpt_oss_power_sft_revised_v3.py \
  --train-file artifacts/traces/out_traces_balanced/sft_traces.train.jsonl \
  --valid-file artifacts/traces/out_traces_balanced/sft_traces.valid.jsonl \
  --model-name unsloth/gemma-4-31B-it \
  --model-revision d722512f8f1e4ef6629c1b24d16d65295c8c945e \
  --output-dir outputs/gemma4_power_agent \
  --max-seq-length 4096 \
  --load-in-16bit \
  --lora-r 16 \
  --lora-alpha 16 \
  --lora-target-scope language_model
```

## Evaluation

### Full evaluation

`submit_eval_v3.sh` runs `eval_sft_agent_gemma_v4.py` on the test split and writes results to `outputs/gemma4_power_agent/`.

Example:

```bash
export MODEL_REVISION=d722512f8f1e4ef6629c1b24d16d65295c8c945e
export ADAPTER_PATH=outputs/gemma4_power_agent/lora
sbatch --constraint=a100 submit_eval_v3.sh
```

Useful overrides:

```bash
export MAX_SAMPLES=100
export MAX_TURNS=6
export MAX_NEW_TOKENS=1024
export GPU_PROFILE=portable
sbatch submit_eval_v3.sh
```

### Smaller smoke evaluation

Build a small balanced subset from the test split with `make_stratified_smoke.py`, then pass that file to `submit_eval_v3.sh`.

```bash
python make_stratified_smoke.py \
  --input artifacts/traces/out_traces_balanced/sft_traces.test.jsonl \
  --output outputs/stratified_smoke.jsonl \
  --per-family 4 \
  --seed 13 \
  --shuffle-output

export TEST_FILE=outputs/stratified_smoke.jsonl
export SMOKE=1
sbatch submit_eval_v3.sh
```

This is the quickest regression check when you only want a small per-family sample instead of the full test set.

## MATPOWER / MCP Usage

The evaluation scripts call the MATPOWER-backed Python tool functions in `mcp_server/matpower_server.py` directly. You do not need to start the HTTP server for the training or evaluation pipeline.

If you want an external MCP-style server for agent experiments, use:

```bash
cd mcp_server
python run_http_server.py
```

## Notes

- `submit_sft_gemma4.sh` and `submit_eval_v3.sh` auto-select defaults based on detected GPU memory and model family.
- Evaluation outputs now default to `outputs/gemma4_power_agent/` to keep the repo root clean.
- `README_DATASET.md` contains dataset-focused notes; this README is intended to document the active root-level Gemma 4 pipeline.

## License

Refer to `LICENSE.md`.
