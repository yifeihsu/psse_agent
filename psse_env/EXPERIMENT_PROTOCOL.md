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

## Targeted DAgger iteration 1

A BC0 checkpoint that fails promotion may be retained as a **learner seed
only**. It is not a release checkpoint. Learner-in-the-loop collection is a
conditional GO only after the runtime rejects every correction that is absent
from the latest same-state context inventory, rejects non-actionable parameter
or topology routes before an executor is called, and circuit-breaks repeated
deterministic non-advancing failures.

The frozen partial-success suite may construct its pre-policy committed repair
through the evaluator's audited full-measurement setup hook. That hook is not
an action argument or policy observation and does not permit an off-inventory
model correction; normal learner and expert execution remains subject to the
same exact context-supported guard.

Run two collection passes with the production release environment (including
the parameter-routing threshold of `1.0`):

1. `beta=0.0` is a diagnostic learner-only pass and is not automatically a
   training corpus.
2. `beta=0.25` through `0.5` is the mixed-policy training pass.

Every input must declare `train` or `dagger_train`, carry an explicit physical
root, and be disjoint from all frozen evaluation roots. Production-eligible
DAgger-1 labels must be rank-one observable-expert targets at learner-created
recovery states; ordinary expert states, hidden-truth labels, and copied
evaluation roots fail closed. Target roughly 300--600 independent recovery
rows, governed by coverage rather than duplication, including unsupported
corrections, post-failure/no-candidate recovery, premature commit and
escalation recovery, multi-measurement safe handoff, and sequential
measurement-plus-parameter recovery.

Build the next view as `D0 union D1`, initially allocating 70--80% to the
eligible BC0 source and 20--30% to learner-recovery rows while retaining the
physical-root and duplicate caps. Run a deterministic targeted tiny-overfit
before the full job. The primary Round-1 run warm-starts from the exact BC0
learner-seed adapter for approximately one epoch at `2e-5` through `5e-5`.

The 13-root failure replay is diagnostic only. It must be written separately
from the frozen suite, marked ineligible for release and training, and may not
replace full evaluation. After any source or DAgger-data change, regenerate
the aggregate and source-bound expert/base evidence before promotion. Keep the
frozen suite and all safety/performance thresholds unchanged.

Model artifacts may be evaluated on any one approved H200, H100, or
high-memory RTX 6000. Candidate and base artifacts are each hardware-attested
independently; their paired comparison does not require the same approved
accelerator class because cluster availability is constrained.

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
