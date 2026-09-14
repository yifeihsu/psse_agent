# Audited IEEE57 training pilot: P0-P2

Implemented and tested in the separate `C:\dw` worktree. The original IEEE14
HPC checkout was not edited. This milestone implements the failure audit,
consistent balanced runtime/teacher configuration, audited canonical SFT export
with replay, and a pure logical-topology transactional adapter. It does not train
model weights or qualify a complete multi-family training release.

## P0: failures and one execution contract

The two historical parameter roots were genuine discrepancies on branch 32
(21-22). Their initial objectives were 330.257 and 372.546, below 424.334, and
their maximum normalized residuals were 3.468 and 3.337, below 4. Their branch
multiplier scores were below 3. Additional parameter-context probes returned no
candidate despite three scans. The parameter optimizer was never called. The
evidence does not establish a missed supported correction; the incorrect resolved
labels must be quarantined. Thresholds were not lowered.

The historical mixed invalid action repeated a correction to meter 471 after
that meter had already been corrected. The provider advertised a post-branch
refinement even though the branch repair preceded the meter repair. Refinement
now requires the observable correction ordering that justifies it. The exact
regression now ends with a protocol-valid audited handoff and zero invalid
actions; it still does not claim strict physical resolution.

Collection, replay, and default IEEE57 evaluation now share
[`ieee57_runtime.py`](../psse_env/dagger/ieee57_runtime.py): alpha=0.05,
maximum normalized-residual threshold 4.0, inclusive OR detection, 40 steps,
four-event controller history, and the production executor-hydrated correction
contract. Live provider settings and actual WLS/verification metrics are checked.
The generic IEEE14 factory's historical defaults remain available.

The first collection attempt exposed a second configuration mismatch: the
generic teacher constructor rejected valid target-only corrections, whereas the
production executor supplies their numerical values. Collection and replay now
use the shared pinned teacher factory. An exact visible-observation regression
prevents that drift. This required no case-ID-specific routing or truth-selected
replacement action.

[Per-episode failure audit](../research/ieee57/training_readiness_failure_audit_20260912.json)
contains the physical/detection/diagnostic/termination evidence and the exact
mixed pre-action observation, bounded history, rejection, and recovery.

## P1: balanced collection, audit, export, and replay

The final run is `output/ieee57_audited_training_pilot_20260913_v4`. Earlier
attempts are retained as incomplete debugging artifacts and are not releases.
Repeated attempts used the same seeded construction; they are not independent
additional populations.

- Fifteen operating parents were assigned to train/validation/test before
  physical generation and noise: five parents per split, one per balanced family.
- Thirty raw physical/control/support records were retained. Fifteen selected
  requested scenarios were evaluated; the other fifteen are retained support
  inputs, not additional evaluated trajectories.
- Parent construction fingerprints exclude seed, label, and split names. Exact
  scenario fingerprints remain separate. All derivatives and counterfactuals
  inherit parent ownership; no cross-parent noiseless-content overlap occurred.
- The teacher fixes each target from the bounded PolicyObservation before the
  private target audit. Private truth can reject a target but cannot replace or
  rerank it. The audit must leave observable execution state unchanged.
- All 164 collected decisions matched the independent evaluator's visible
  observations and actions. Every requested root remains in evaluation.
- Sixty-two TRAIN targets passed their observable, private, protocol, schema,
  and realizability checks and were exported. No held-out parent entered SFT.
- All 62 exported targets replayed with fresh controller IDs, fresh aliases,
  repeated private target checks, matching post-action states, and exact terminal
  outcomes. Raw prefixes are replayed even where supervision is quarantined.

The held-out trajectories contained four quarantined targets: three unsupported
measurement refinements and one incorrect parameter-case finalization. There
were no quarantined targets in the five selected training trajectories. The
quarantine machinery is also tested on failed training trajectories, preserving
their independently valid earlier targets.

| Family | Evaluated roots | Audited task successes, including qualified handoffs | Strict resolved |
|---|---:|---:|---:|
| Clean | 3 | 3 | 3 |
| Single measurement | 3 | 3 | 0 |
| Multiple measurements | 3 | 3 | 0 |
| Parameter | 3 | 2 | 0 |
| Measurement + parameter | 3 | 1 | 0 |
| Total | 15 | 12 | 3 |

All 15 roots have known healthy-component preservation, with preservation on all
15. There were zero invalid actions, zero reported false commits, one false
finalization, and zero infrastructure errors. A task success via handoff is not
an autonomous resolved episode.

After filtering, the training view contains seven measurement-correction targets,
one parameter-correction target, eight commits, four handoffs, one clean
finalization, and the associated WLS/context/acquisition requests. The mixed
training root does not demonstrate a complete parameter-plus-measurement repair.
One parent per family and one parameter-repair target are insufficient for a
qualified five-family corpus. This is a mechanics pilot, not a sealed final test
or a diagnostic-accuracy benchmark.

### Replay and publication safeguards

Replay exposed an existing export portability issue: history character budgets
were applied before controller IDs were aliased, so longer fresh IDs could remove
an entire history event. The new opt-in `alias_before_compaction` view normalizes
mapping order and aliases IDs before compaction. Current active/candidate
references take precedence over historical ones. The default legacy view is
unchanged: an independent comparison found exact agreement with the previous
implementation on all 164 recorded views and bindings. The opt-in views were
also invariant on all 164 observations under longer IDs and reversed mapping
order.

The pilot freezes participating production source files across generation,
collection, evaluation, and replay. It compares persisted artifacts with the
in-memory evidence that passed the gates, publishes the final SFT filename
atomically, and writes the successful manifest last. A failed attempt cannot
leave a qualified training filename or `complete=true`.

Reproduce from the repository root with a fresh output directory:

```sh
python scripts/run_ieee57_training_pilot.py --output-dir output/ieee57_pilot_new --seed 20260913 --per-family 1
```

The resulting `train.canonical.jsonl` is a canonical tool-call SFT mechanics
dataset, with parent and audit provenance outside model-visible messages. No
model fitting, tokenization-specific training gate, or learned-policy evaluation
was performed.

## P2: pure logical-topology integration

The opt-in [`logical_adapter.py`](../psse_env/dagger/logical_adapter.py) reuses the
real transactional environment and state store. Its separate canonical protocol
supports `correct_logical_topology_from_context` with a candidate ID, certificate
hash, and an exact map of desired statuses. A certified pair is one atomic
transaction, not two independently accepted single changes.

Verification and commit retain all 530 raw measurements, their identities and
covariance, equipment parameters, and prior statuses. Certificates, current
context/settings, candidate files, and evidence hashes are rechecked. Private
audits use the logical device identities, including couplers, and inspect each
accepted intervention. Missing WLS, stale scope, truncated searches, altered
certificates/data/candidate files, and unsupported actions cannot authorize a
correction or false completion.

The seven-fixture protocol pilot produced six strict logical resolutions and one
inconclusive outcome. All actions succeeded. Searches covered all 93 single
alternatives and one predeclared pair for the paired fixtures. These are noiseless
protocol fixtures from three physical parents, not independent training data or
full multi-error identification coverage. Logical resolution is explicitly a
fixed-evidence, connected-estimation result within the declared search scope;
it is not an AC-PF operating-safety certificate.

```sh
python -m psse_env.dagger.logical_pilot --output output/logical_adapter_new
```

The default legacy `env_kwargs()` guard and raw-section mixed-error guards remain
enabled. The balanced SFT collector is not silently repurposed for 530-channel
topology data. A logical training population and its audited export remain a
subsequent dataset milestone, using the adapter now implemented.

## Retrievable evidence

Full trajectories and numerical inputs are now versioned as compressed archives,
with per-file SHA-256 manifests, rather than available only at local output paths.
Archives preserve original path provenance; regenerate a new run from source for
execution in a differently located checkout.

- [Balanced pilot summary and episode audits](../research/ieee57/audited_balanced_pilot_20260913.json)
- [All 164 action-level audit decisions](../research/ieee57/balanced_target_audit_20260913.json)
- [Independent view/alias compatibility audit](../research/ieee57/model_view_compatibility_20260913.json)
- [Final test and source validation](../research/ieee57/final_validation_20260913.json)
- [Balanced full inputs, raw targets, evaluation, SFT and replay](../research/ieee57/evidence/audited_balanced_pilot_20260913.tar.gz)
- [Balanced archive inventory/hashes](../research/ieee57/evidence/audited_balanced_pilot_20260913.tar.gz.manifest.json)
- [Logical pilot receipt](../research/ieee57/logical_adapter_pilot_20260913.json)
- [Logical full scenarios and traces](../research/ieee57/evidence/logical_adapter_pilot_20260913.tar.gz)
- [Logical archive inventory/hashes](../research/ieee57/evidence/logical_adapter_pilot_20260913.tar.gz.manifest.json)
- [Historical ten-root full inputs and evaluation](../research/ieee57/evidence/training_readiness_full_20260912.tar.gz)

## Tests and remaining scope

The final combined logical/balanced/provider/recovery/export suite passed **353
tests and 79 subtests**. The
independent 164-state compatibility and ID-invariance checks passed, as did all
end-to-end pilot publication/replay gates.

P3 mixed logical corrections still require nuisance-aware partial acceptance and
raw-section measurement/parameter routes. P4 still requires phase acquisition and
supervision integration, including an observable acquisition policy that can run
after a non-alarming WLS result. NLM ranking remains deferred. Additional
independent parents, coverage after filtering, and audited recovery/stopping
examples are required before scaling or claiming a complete training release.
