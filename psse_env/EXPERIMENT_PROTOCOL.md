# Recovery-aware DAgger experiment protocol

All expert, injected-error, and recovery branches from one
`root_scenario_id` must remain in the same split. Use
`dagger.grouped_scenario_split`; never split individual branch rows.

Hold out combinations across network case, error family, component location,
fault magnitude, triple-error composition, and tool-failure pattern.

The required evaluation suites are:

1. standard physical success;
2. forced first-error recovery;
3. partial-success retention;
4. invalid-action recovery;
5. efficiency and tool regret.

Report final physical correctness, healthy-component preservation,
forced-error recovery, partial-success retention, false rollback, false
commit, false finalization, loop rate, WLS/tool calls, and tool-call regret.
Exact sequence match is secondary.

Do not start a real DAgger run unless policy leakage, no-op invalid actions,
episode-local references, mandatory verification, candidate-oracle wiring,
partial-versus-reject labeling, unknown-fault handling, updated-history
collection, and best-checkpoint selection all pass.

Do not start AggreVaTe-lite until seeded branches are reproducible, recovery
cost is tested, top-L proposal recall is measured, and DAgger recovery has a
stable baseline. Do not start reward-model RL until verifier calibration and
transaction semantics have been validated.

## Gemma 4 SFT launch gate

Generate and audit the bundled production-mode pilot from the archive root:

```bash
python -m psse_env.examples.generate_sft_pilot \
  --output-dir psse_env/examples/sft_pilot

python -m psse_env.sft gate \
  --model unsloth/gemma-4-31B-it \
  --revision 8a796db4df380b178065ed910849477ff0e99c87 \
  --train psse_env/examples/sft_pilot/pilot.train.jsonl \
  --validation psse_env/examples/sft_pilot/pilot.validation.jsonl \
  --test psse_env/examples/sft_pilot/pilot.test.jsonl
```

The gate requires 32-128 production-tagged rows, disjoint root groups, valid
row-level JSON schemas, dictionary-valued arguments, exact processor rendering,
assistant-only masks, nonzero supervision, no target truncation, and tool-call
round trips. The generator separately requires observable provider declarations,
zero hidden-provenance leakage, zero teacher conflicts at the configured
tolerance, and target-aware state-class consistency.

The bundled 90-row dataset is a tokenizer and training-stack pilot only. It has
no rejected-candidate, partial-commit, invalid-precondition, or loop examples,
and validation/test each contain one root group. Before full 31B SFT, generate a
recovery-balanced aggregate, run the exact 31B forward/backward and tiny-overfit
gates on HPC, and pass a short root-group-held-out recovery evaluation.
