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

1. `beta=0.0` is a diagnostic learner-only pass over the primary roots only
   and is always training-ineligible.
2. `beta=0.25` through `0.5` is the mixed-policy training pass.

Every input must declare `train` or `dagger_train`, carry an explicit physical
root, and be disjoint from all frozen evaluation roots. Production-eligible
DAgger-1 labels must be rank-one observable-expert targets at learner-created
recovery states and must pass a post-target, privileged physical-truth audit;
ordinary expert states, hidden-truth labels, copied evaluation roots, and a
truth-inconsistent correction/commit/rollback/finalize/handoff target fail
closed. The teacher target and rank-one proof are computed from the exact
`PolicyObservation` before `OracleState` is materialized; private truth can only
quarantine or score that already-fixed target. The teacher receives only that
observation's bounded `history_window`, never the collector's fuller private
transition history. The audit record is a fixed
low-bandwidth boolean/reason-code object in
non-model metadata and can quarantine a target only after the observable
teacher has fixed it. Training collection must load one absolute local adapter
directory whose inspected tree digest is the supplied 64-hex model revision.
The collection manifest, final aggregate, and `STAGE=round1`
`INITIAL_ADAPTER_REVISION` must all bind that exact learner-seed digest.

Do not equate a post-correction WLS pass with physical release finality. The
same public residual pattern can occur after both a correct repair and a repair
that masks a remaining fault. Production therefore persists a policy-visible
confirmation obligation after an accepted correction, investigates the active
state with observable context, and uses an audited operator handoff when the
remaining budget or autonomous evidence is exhausted. The private truth audit
may reject a proposed finalization, but it may not rewrite the observable
teacher target into a privileged resolution label.

The safety-first behavior is bound to
`bc0_observable_sequential_handoff_v2` and
`bc0-observable-handoff-expert-v2`; DAgger-1 rows bind
`dagger1_observable_recovery_handoff_v2`. Release policy v3 preserves every
pinned numeric floor and ceiling but makes the evidence category explicit:
audited completion is strict physical resolution or an exact state-bound
post-correction controller handoff with a passing separate private completion
audit. The production label remains `operator_escalation`. Partial, HIF,
budget, or otherwise generic handoffs remain unqualified escalations and do
not enter the numerator; failure to meet the audited-completion or
unqualified-escalation bounds remains an explicit release NO-GO.

The Round-0 manifest must bind every strict audit and completion assessment to
an independently persisted final transition/store anchor and a zero-false
lifecycle audit. Its SHA-256 is a required D0 input binding for scenario,
holdout, collection, and Round-1 aggregate builders; transport packages retain
an external archive checksum as the trust anchor for the unpacked evidence.

Build one finite, deterministic candidate inventory by requesting 108
measurement-plus-parameter, 176 multi-measurement, and 48 parameter candidates.
From the eligible fresh roots, select the 48/48/24 primary training allocation,
the exact 12/12/6 development allocation, and a finite reserve of 60 additional
measurement-plus-parameter plus 31 additional multi-measurement roots. There is
no parameter reserve. The development multi-measurement allocation must hold
back exactly three roots for each measurement-error cardinality 2, 3, 4, and 5.
All allocations remain disjoint from D0 and the frozen evaluation suite.

This 12-root mixed-family reserve buffer is an explicitly named and hashed
fresh-root top-up in response to the exhausted prior schedule: seven of the
nine independently supported `post_failure_no_candidate` roots came from the
measurement-plus-parameter family. The prior 199-root allocation is hash-bound
as the predecessor set, and the top-up must be disjoint from it and from the
development reservation. It increases natural physical-root opportunity
without manufacturing a failure, counting a replica as independent support, or
weakening the 10-root support floor; the finite collector can still terminate
NO-GO.

The mixed-policy collector uses a root-local deterministic beta seed, so an
episode's expert/learner choices do not change when earlier roots terminate or
when a reserve is unused. It executes whole episodes in the declared primary,
reserve, and repeat schedule, with at most two replicas of a
measurement-plus-parameter root, three replicas of a multi-measurement root,
and one replica of a parameter root. This is pure DAgger collection: reserve
and repeat episodes follow the same learner/expert mixture and do not inject a
scripted failure or teacher action.
Full reserve exhaustion is therefore finite at 477 episodes. The collector
stops after the first passing whole-episode batch checkpoint. Because the
strict truth-audit gate requires zero quarantined targets and all visited rows
remain in the cumulative audit, the first whole-batch checkpoint containing a
quarantined target instead terminates immediately with an explicit irreversible
NO-GO; later batches cannot repair that failure.

Every executed episode must end with one explicit disposition:
`resolved`, `operator_escalation`, or `horizon_truncated`. Horizon truncation is
a terminal collection disposition, not proof that the physical task was
resolved. Its learner-created hard states can still provide valid DAgger
supervision when their individual labels pass the observable rank-one and
offline truth audits; horizon truncation alone is not a collection-gate
failure.

The deterministic selector must publish 300--600 independent recovery rows,
governed by coverage rather than duplication, including unsupported
corrections, post-failure/no-candidate recovery, premature commit and
escalation recovery, multi-measurement safe handoff, and sequential
measurement-plus-parameter recovery. Each of the ten predeclared targeted
state cells requires at least five distinct physical roots.
Unsupported-correction, post-failure/no-candidate, measurement-parameter
sequencing, and multi-measurement handoff each require at least ten; premature
commit and premature escalation each require five. Repeated rows from one root
never count as independent support. The selector preserves these rare-root
floors, enforces the 600-row maximum, and treats Round-1 replay capacity under
the duplicate and per-root caps as a strict GO condition.

After every declared whole-episode batch, stop at the first deterministic
strict GO. The complete visited ledger retains every visited row, including
all safe candidates that the bounded selector did not choose. The published D1
training JSONL is the selector's deterministic at-most-600-row subset, not an
alias for the visited ledger. Aggregate ingestion must recompute and verify the
same selection from the bound ledger and manifest; it must not relabel or
promote a discarded candidate.

A strict mixed-policy collection must name a separate, write-once failed-run
directory before launch. If the finite primary/reserve/repeat schedule is
exhausted before strict GO, publish only an atomic checksummed,
reserve-exhausted NO-GO bundle there and exit nonzero. Do not create the
requested D1 production JSONL, all-row ledger, or manifest on that path; the
diagnostic bundle retains the complete visited evidence but is explicitly
training- and aggregate-ineligible and exists only to make the next correction
evidence-based.

Build the next view as `D0 union D1`, initially allocating 70--80% to the
eligible BC0 source and 20--30% to learner-recovery rows while retaining the
physical-root and duplicate caps. Before sampling, report the requested and
largest feasible view under both caps; insufficient replay capacity fails
closed. Preserve immutable natural D0, the complete D1 visited ledger, the
selected D1 training view, and the combined source views. Final ingestion must
recompute the deterministic D1 selector, recovery/class/root and replay-capacity
audits, exact realizability on the natural union and balanced view, and
approximate realizability overall and by family, state class, and explicit D1
recovery stratum with nonzero comparison coverage. Persist the complete reports
in preflight and bind their hashes in generation provenance. Run a
deterministic targeted tiny-overfit before the full job. The primary Round-1
run warm-starts from the exact BC0 learner-seed adapter for approximately one
epoch at `2e-5` through `5e-5`.

The 120 primary D1 training roots contain 48 measurement-plus-parameter, 48
multi-measurement, and 24 parameter roots. The deterministic 30-root
development suite contains 12, 12, and 6 respectively from the same train
source partition, with the multi-measurement roots stratified three per
cardinality 2--5. Development must remain disjoint from primary and reserve D1
roots, D0, and all 115 frozen roots. It is diagnostic model-selection evidence
only: it never enters SFT and never counts as promotion evidence. Use it for
closed-loop learning-rate and replay-share checks before the one-time
frozen-suite promotion evaluation. The present evaluator can compare outcomes
on these roots but cannot certify expert-defined recovery-stratum opportunity
coverage because it persists only observation hashes and learner actions. The
holdout is therefore not stratum-qualified until diagnostic-only teacher-target
instrumentation and a binding post-evaluation coverage audit are added.

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
