# Remaining comparison-teacher gap: review and minimal interface repair

Reviewed 2026-10-23 against `e7a65e2f481a38d209becfd40c51dd32c8d8b77b` on
`codex/classifier-triage-20261004`. GitHub's compare endpoint reported
`identical`, zero commits ahead, when compared with that branch. No later
fix was present at review time. The completed 885-step DAgger round must
not be launched again.

## Evidence and its limits

Sources read: `docs/rl_experiments_20261014.md`, sections 12–13;
`docs/decision_audit_20261019/compare_dev73_summary.{md,json}`; the current
expert, process gate, transactional environment, model observation exporter,
paired evaluator, and comparison-cell launch/configuration files.

Section 13 reports 197/209 student successes versus 201/209 expert
successes, with 6 invalid actions, 9 loop episodes and 56 comparisons.
The four expert-only roots are a gap between policies, not all failures:
the student has 12 failures and the expert 8. The paired audit below also
checks student-only successes instead of assuming the success sets nest.

The repository contains the earlier expert comparison summaries, but not
the completed cell's full student/expert evaluation JSON, original hard
suite, or trained adapter. Those files are recorded under the NYU HPC
cell. This environment has no accessible CUDA device or installed
PyTorch/Transformers model stack. Therefore this review has **not** replayed
the 209 roots, identified the two mimic root IDs from raw traces, or
measured an accuracy/latency improvement. No replacement draw was generated.

## The parameter failures are an interface ambiguity

| Root | Observable ratio | Context top line | Comparison request | Observable verdict order |
| --- | ---: | ---: | --- | --- |
| `r0_3ad37c833dc5` | 1.3036151181225657 | 3 | 3, 6, 4 | 6, 3, 4 |
| `r0_8013605195d2` | 1.303707550312555 | 3 | 3, 6, 4 | 6, 3, 4 |

The context's `dominant=true` is correct under its **1.2 admission ratio**.
It is insufficient to decide whether the comparison expert skips a test:
`ExpertPolicyOracle._parameter_comparison_wanted` compares a non-dominant
ranking, or a dominant ranking whose numeric ratio is **strictly below 1.5**.
At exactly 1.5 a dominant ranking takes the direct route. At least two
supported lines, a fresh active-state context, no existing comparison,
and the normal process/lifecycle requirements still apply.

Before this patch, `protocol_bridge.py` only told the student to compare
when the ranking "does not clearly dominate." The numeric rule was absent
from that description. The exporter preserves the observable ratio and
flag, so this is a missing instruction, not a missing numeric input.
Section 13 records only 14 comparison demonstrations with a dominant flag
and a ratio in [1.2, 1.5), against about 100 direct corrections above the
band. The student's observed decisions are consistent with a boolean
shortcut; they do not prove a general inability to compare numbers.

The repair moves the unchanged default `1.5` into `psse_env.actions` and
uses it in both the expert and tool description. The description explicitly
distinguishes admission from comparison, spells out the strict inequality,
and tells the student to reuse a fresh comparison. The context's 1.2 flag,
verdict ordering, correction providers, transaction rules and verifier are
unchanged. The expert module still exports the imported constant for
existing callers. A custom expert band remains an explicit experimental
override; this schema describes the default comparison-teacher cell.

This rule names no faulty line, root, or scenario family. It was selected
from the training audit before these development results. Line selection
still uses the isolated branches' observable verification verdicts. The
test fixture has synthetic physical truth, as existing environment tests
do; neither tool description nor policy payload acquires that truth.

If the description recovers precisely these two roots without regression,
the score would be 199/209. **That is a hypothesis, not a measured result.**
Perfectly matching the teacher's trigger would add four comparison calls
(56 to 60), not necessarily just two: the student also skipped comparisons
on two dominant-stratum roots it already solved. Comparison cost must include
the internal correction and WLS calls, not only the one charged policy action.

## Why the mimic loops remain plausible despite durable state

The following mechanisms are confirmed in code:

1. `transactional_env.py` records `request_attempted`, acquisition status,
   `nlm_attempted`, `nlm_classification` and any observed HIF refutation in
   the state-bound acquisition ledger. A process-refused call does not mark
   an acquisition/diagnosis as performed. Refresh replaces its acquisition
   record; state changes retire or explicitly carry records under existing rules.
2. `prepare_model_policy_observation` preserves those fields even with no
   recent history. The research environment supplies a four-event history;
   the adapter's eight-event ceiling cannot restore already-dropped events.
   `nlm_localization` is controller-only, but the attempted/classification
   fields are not. Adding generic memory is not yet a justified fix.
3. The expert's `_rank_and_filter` rejects already-seen domain action
   signatures, and its diagnostic ladder reads the durable ledger. The
   learner's process gate checks whether a diagnostic is admissible; it
   does not generally reject an already-answered spectra request or NLM
   diagnosis. `current_suspicion` intentionally retains admission for a
   stream whose original suspicion was subsequently refuted.
4. The evaluator recognizes repeated nonadvancing actions as loops. Its
   deterministic-failure breaker does not eliminate successful but
   redundant read-only calls. Merely rejecting more calls can increase the
   invalid-action count without teaching a useful next action.

A CPU fixture under `suspicion_gated_diagnostics` confirmed that, with a
valid unexplained screen, balanced-three-phase NLM classification, both
attempt flags already set and both action signatures already tried, the
process gate still permits both spectra and NLM. The same fixture's ledger
survives model compaction. This establishes a mechanism, **not** the exact
causal sequence of the two unavailable student traces.

Before changing termination, inspect the first redundant action in each
expert-only mimic root: current state and acquisition status, original
request success versus refusal, remaining supported measurement corrections,
open candidate lifecycle, rejected targets, and the expert's continuation.
Clean phase/spectral evidence does not establish that the balanced meters
are correct. The student may need another supported measurement correction,
candidate verification/commit, or a valid operator request. It must not be
automatically finalized or declared successful merely to stop a loop.

## Three interventions, kept separate

| Intervention | Category | Priority and falsifiable check |
| --- | --- | --- |
| State the existing 1.5 rule in the tool description | Information interface; no weight update | Implemented. Evaluate the completed adapter on all 209 roots; inspect both 1.30 decisions and all prior 197 successes. |
| Add boundary demonstrations | Additional learning | Only if the explicit interface still leaves boundary errors. Collect on the existing 256 training roots or a new parent-disjoint training pool, include dominant examples below 1.5 and direct corrections at/above it, and mix preservation examples. Keep the 209 development roots and their physical parents out of SFT/DAgger rows. Do not fabricate labels by just editing a ratio in an otherwise inconsistent observation. |
| Make diagnostic completion and closure clearer | Interface first; optional later learning or controller rule | First inspect the saved mimic traces. A short model-visible summary/prompt can reuse the existing ledger without new truth. If exact same-acquisition calls persist, consider a separately evaluated repeat guard plus a legal recovery hint. Retry after a real acquisition/state change or an earlier gate refusal must remain possible. A general handoff still has to pass existing exhaustion and physical audits. |

For a loop intervention, scope repeats to the state and acquisition, not to
the whole episode. Reopening phasors after a commit can be legitimate.
Do not let a guard silently invoke the expert to choose the next action;
that would change the policy being evaluated. Do not weaken success,
healthy-component preservation, false-commit, or handoff validation.

Run ablations in order: current completed adapter/old interface, the same
adapter/explicit band, that adapter/band plus a separately specified
diagnostic-state instruction, and only then an optional new small DAgger
update. The previously trained 885-step adapter is the starting point.
If development failures guide an additional training design, the 209 roots
remain a development set; reserve a fresh untouched set for a final paper claim.

## Code and CPU validation

- `psse_env/actions.py`, `psse_env/oracle/expert_policy.py`,
  `psse_env/dagger/protocol_bridge.py`: shared default and explicit tool text.
- `psse_env/test_candidate_comparison.py`: boundary decisions at 1.19, 1.2,
  1.30, 1.499999, 1.5 and 1.500001; model-visible ratio/flag and schema.
- `research/rl/compare_interface_audit.py`: offline paired root audit; counts
  gains, losses, expert-only/student-only roots, invalid actions, loops,
  calls by tool, comparison internal work and costs on common successes.
  It uses the saved `truth_audited_task_success` flag and refuses mismatched
  root sets, episode seeds or budgets. It does not re-score a handoff.
- `research/rl/evaluate_saved_comparison.py`: evaluation only, using the
  completed r1 adapter and saved model/environment settings. It refuses a
  different root order or altered suite and records code, suite and adapter
  hashes. A mismatched output manifest cannot reuse old checkpoints.
  Its student output is named `bc0_eval.json` by the reused evaluator API;
  the weights are the **completed r1 adapter**, not the old frozen baseline.

Local validation: **86 passed, 2 skipped**. The two skips require the older
`output/classifier_triage_20261004/suites/development.json`, absent here.
Numerical test dependencies were installed in an isolated CPU environment
(including PYPOWER 5.1.19 and OpenDSSDirect.py 0.9.4). The evaluation driver's
preflight and manifest protections were tested with explicitly synthetic
metadata; no GPU inference was exercised. Two third-party deprecation
warnings were emitted.

```bash
python -m pytest -q \
  research/rl/test_evaluate_saved_comparison.py \
  research/rl/test_compare_interface_audit.py \
  psse_env/test_candidate_comparison.py \
  psse_env/dagger/test_protocol_bridge.py \
  psse_env/dagger/test_scada_policy_tools.py \
  psse_env/oracle/test_refused_diagnostics.py \
  psse_env/oracle/test_unexplained_handoff.py \
  psse_env/oracle/test_contract_20261007.py
```

## HPC commands: inspect first, then evaluate only

Use the original cell's assets and Python, with the review code in a separate
checkout. Fetching this branch and adding a worktree can be done on a login
node; run inference within an appropriate GPU allocation. These commands
are provided for execution there; they were not executed on HPC in this review.

```bash
CELL=/scratch/yx3882/research_full_pipeline_20261021_compare
REVIEW_SRC="$CELL/interface_review_source_20261023"
git -C "$CELL/source" fetch origin codex/compare-interface-audit-20261023
git -C "$CELL/source" worktree add --detach "$REVIEW_SRC" FETCH_HEAD
source "$CELL/pipeline.env"
pipeline_environment
export PYTHONPATH="$REVIEW_SRC"
cd "$REVIEW_SRC"
COLLECT="$CELL/out/r1/collection"
CHECK="$CELL/out/interface_band_20261023"

# CPU: obtain the four actual expert-only IDs and inspect their paired traces.
"$PY" -m research.rl.compare_interface_audit \
  --baseline "$COLLECT/evaluation/r1_eval.json" \
  --candidate "$COLLECT/evaluation/r1_eval.json" \
  --expert "$COLLECT/evaluation/expert_eval.json" \
  --output "$CELL/out/interface_audit_20261023/before.json"

# CPU preflight: require the saved 209 roots, completed weights and fresh output.
"$PY" -m research.rl.evaluate_saved_comparison \
  --run-report "$COLLECT/research_run_report.json" \
  --suite "$COLLECT/development_scenarios.json" \
  --output-dir "$CHECK" --dry-run

# GPU allocation: inference and expert evaluation only; no collection or training.
"$PY" -m research.rl.evaluate_saved_comparison \
  --run-report "$COLLECT/research_run_report.json" \
  --suite "$COLLECT/development_scenarios.json" \
  --output-dir "$CHECK" --workers 1

# CPU: the baseline is the completed 197/209 adapter's original evaluation.
"$PY" -m research.rl.compare_interface_audit \
  --baseline "$COLLECT/evaluation/r1_eval.json" \
  --candidate "$CHECK/evaluation/bc0_eval.json" \
  --expert "$CHECK/evaluation/expert_eval.json" \
  --output "$CHECK/paired_audit.json"
```

Check that the saved baseline is 197 and the unchanged teacher remains 201;
then require zero regressed roots among the original 197, report both named
parameter decisions, all 12 original failures, all 9 loop episodes, invalid
actions, per-tool calls and internal comparison work. Do not accept a net
gain alone as evidence of no regression. If the baseline cannot be reproduced
under the pinned environment, resolve that discrepancy before attributing
any change to the description. A prompt-only change can affect other families,
as section 13's zero-shot tool-listing experiment already showed.
