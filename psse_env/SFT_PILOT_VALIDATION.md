# DAgger Gemma 4 SFT pilot validation

> Historical pilot evidence only. The committed JSON reports in
> `psse_env/examples/sft_pilot/` predate the expanded diagnostic registry and
> do not carry source/schema/exporter provenance. They must not be accepted as
> current release evidence. Regenerate the exact probe and live dataset gate
> from the commit being evaluated.

## Decision

**CURRENT NO-GO.** The 90-row results below are a historical record only and
are release-ineligible against the current code and expanded tool registry.
They do not authorize a pilot or full 31B SFT run. Regenerate a provenance-bound
canonical aggregate and rerun the exact processor/template/mask gate from a
clean commit before any optimizer stage. The resulting checkpoint must then
pass a real forward/backward step, tiny-overfit gate, and short
physical-root-held-out recovery evaluation on suitable HPC hardware.

## Historical pilot gate results

| Required gate | Result |
| --- | --- |
| Row-level JSON tool schemas | pass, 13 complete schemas on every row |
| Dictionary-valued assistant arguments | pass, 90/90 rows |
| Exact 31B `apply_chat_template()` | pass at pinned revision on 90/90 rows |
| Assistant-only token mask | pass, prompt labels `-100` token by token |
| Nonzero supervised target | pass, zero failing rows |
| Target/prompt truncation at 4096 | pass, zero/zero rows |
| Privileged and semantic provenance audit | pass, hidden/oracle sources rejected |
| Observation-to-action conflict audit | pass, 0.0 conflict rate across 38 canonical observations |
| Local controller aliases and reverse binding | pass, including one-character episode-ID regression |
| Strict production provider configuration | pass with explicitly approved deterministic pilot adapters |
| Grouped split before export | pass, 13/1/1 disjoint train/validation/test root groups |
| Target-aware class audit | pass, zero mismatches |
| Exact local E2B one-batch | pass, finite gradients and parameter change |
| Exact local E2B tiny overfit | pass, `14.1112756729 -> 0.0005142066` in 20 steps |
| Greedy generated tool-call parse | pass, `run_wls` with dictionary arguments |

Production collection now fails closed when domain context labels lack an
observable family signature, corrections lack fresh observable context,
or correction arguments do not exactly match the context provider's bounded
`supported_corrections`; commit/rollback labels require decision evidence, and
finalization requires observable terminal evidence. Hidden terminal flags and
private hints no longer select context-routing labels.

## Pilot evidence

- Seed: `20260715`
- Roots: 15
- Rows: 90 raw/exported; 78 train, 6 validation, 6 test
- Native chat audit: 90/90
- Preferred actions: 30 WLS, 15 context, 15 correction, 15 commit, 15 finalize
- State classes: 60 clean-successful, 15 accepted-final-commit, 15 terminal-decision
- Exact 31B length maximum: 2,440 tokens
- Exact all-tool target probe: 13/13 on both pinned E2B and 31B processors

Pilot file hashes:

```text
9b0026d33e06762371192a804d95fd152d3bcc07820885cbd52b6b3c6329061c  pilot.train.jsonl
ce0070d6dbaae0b4a2f358f9ca38a4c528bfaad5c760391e35c6c22e5eca3bfd  pilot.validation.jsonl
29ace51178355195d59fe5d94ef0b12c2d4d3efb784ca902eb21edef84c4c8dd  pilot.test.jsonl
```

Machine-readable reports:

- `examples/sft_pilot/pilot.preflight.json`
- `examples/sft_pilot/gemma4_31b_tokenizer_gate.json`
- `examples/sft_pilot/gemma4_e2b_qlora_smoke.json`
- `examples/sft_pilot/all_tools_exact_processor_probe.json`

## Reproduce from archive root

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -q

python -m psse_env.examples.generate_sft_pilot \
  --output-dir psse_env/examples/sft_pilot

python -m psse_env.sft gate \
  --model unsloth/gemma-4-31B-it \
  --revision 8a796db4df380b178065ed910849477ff0e99c87 \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl \
  --test psse_env/examples/sft_pilot/pilot.test.jsonl

python -m psse_env.sft smoke \
  --mode one-batch \
  --model unsloth/gemma-4-E2B-it \
  --revision f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl

python -m psse_env.sft smoke \
  --mode tiny-overfit --tiny-overfit-steps 20 --learning-rate 0.001 \
  --model unsloth/gemma-4-E2B-it \
  --revision f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl
```

The exact processor/model commands require the pinned checkpoints in the local
Hugging Face cache or an explicitly download-enabled run.

The `0.001` value above reproduces the archived E2B report and its displayed
loss trajectory. It is not the current 31B release setting. On H200/H100, the
reviewed launcher defaults the current tiny-overfit stage to `0.0001`:

```bash
TINY_OVERFIT_LR=0.0001 STAGE=tiny-overfit \
  sbatch submit_dagger_sft_round0.sh
```

## Remaining full-run work

1. Freeze the suite, four factories, policy, and hashes in one commit.
2. Regenerate the release aggregate from that commit; require
   `release_eligible=true`, `aggregate.train_view.jsonl`, and a legitimate row
   count inside the launcher bounds.
3. Run and validate the CPU observable-expert baseline
   (`EVALUATION_MODE=expert` on `submit_dagger_release_eval.sh`, or the
   `evaluate_release`/`validate_evaluation` CLIs directly).
4. Run and validate the exact base-Gemma GPU baseline before any optimizer
   stage with `submit_dagger_release_eval.sh`; supply the externally reviewed
   freeze commit through `REVIEWED_SOURCE_COMMIT`.
5. Execute `gate -> one-batch -> tiny-overfit -> round0` with `afterok`
   dependencies, passing that same reviewed commit to every stage.
6. Evaluate every produced `output/lora` adapter on the frozen suite with the
   release-evaluation launcher. Its content digest names the evidence artifact,
   and its output guard protects the persisted base reference.
7. Run `STAGE=checkpoint-gate` against each persisted checkpoint evaluation and
   the paired base artifact. This stage validates an existing evaluation
   artifact; it does not run closed-loop evaluation itself.
