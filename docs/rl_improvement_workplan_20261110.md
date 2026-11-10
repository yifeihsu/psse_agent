# Draft workplan for RL improvement in PSSE Agent

This workplan aims to make the learner solve more compound-error cases than the rule expert while preserving correct diagnoses and using tools efficiently. The proposed order is to verify the approved updater repair, restore complete observable tool history, identify successful recovery routes, and then test history and decision credit in short, matched RL runs.

The first training comparison should change the history available to the agent while holding the repaired updater and current learning objective fixed. Continuation-based credit is the next intervention if the diagnostics show that better choices can produce successful recoveries. Pilot sizes and promotion thresholds below are proposed engineering choices.

Draft dated 10 November 2026. Code reference: branch `claude/rl-lifecycle-evidence-20261106`, commit `4b14d5e`.

## Goal and starting point

The primary outcome is audited episode success under the existing evaluation contract. Report clean success and preservation of healthy components alongside it. Efficiency matters after successful recovery has been established.

The supplied 9 November 03:11Z development snapshot was:

| Policy | Solved out of 300 |
| --- | ---: |
| Rule expert | 188 |
| Audit-assisted expert | 200 |
| S0 cloned start in bf16 | 182 |
| S1 after iteration 0 | 180 |

In that snapshot, S1 gained no solves over S0 and lost two; 265 tool sequences were identical. Beating the expert on this same subset would require at least seven net gains from S0. The audit-assisted score is a restricted diagnostic reference based on its intervention and continuation policy, rather than a general achievable upper bound. [Audit reader definition](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/research/compound_errors/commit_reader_loop.py#L41-L47)

Before the next run, append every newly completed checkpoint evaluation to a baseline register, with its checkpoint, evaluator, precision, observation mode, and configuration. The historical snapshot does not establish a current S2 or S3 score.

Continue from the cloned adapter through learner-generated RL experience. Preserve the decision to avoid another compound-error DAgger round. Keep the original PESGM results and the new research variants separately identified. Retain the established treatment of isolated-line parameter audit as N/A and the optional isolated-line declaration.

## Delivery stages

Owners below are roles to assign; no named person or job is assigned by this draft.

| Stage | Main deliverable | Lead role | Gate to proceed |
| --- | --- | --- | --- |
| 0 Update integrity | Verified updater, calibrated update size, reproducible baseline manifest | RL implementation | Numerical and resume checks pass; intended choices move without a general decline bias |
| 1 Observable history | Versioned history mode and token audit | Environment and policy interface | All eligible earlier outputs survive; state binding and input limits are explicit |
| 2 Recovery diagnosis | Paired decision and continuation report | RL evaluation | Successful legal recoveries exist and observable evidence helps distinguish them |
| 3 History pilot | Matched short runs A and B | RL experiments | Positive paired net solves with acceptable clean-success and preservation results |
| 4 Decision credit | Conditional runs C and D | RL implementation and experiments | Continuation credit produces additional completed recoveries |
| 5 Confirmation | Three-seed comparison and untouched evaluation | Evaluation | Reproducible gains over the expert with uncertainty reported |

Stages 0 and 1 can progress in parallel after their shared observation and checkpoint contracts are agreed. The continuation harness can be prepared concurrently. Training begins after the relevant implementation gates pass.

## Stage 0 Verify and calibrate the updater

The local updater repair has already been approved in the recent project discussion. Its completed implementation and tests should be the first dependency of the new experiment series.

The inspected implementation partitions one pass into two optimizer chunks and constructs AdamW unless an optimizer is supplied. Whole trajectories remain together, while samples from the same root can be assigned to different chunks. [Update implementation](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/rl/update.py#L501-L605)

Treat the repair as a defined algorithmic change whose effect must be measured. Splitting samples after complete-group advantages have been calculated is mathematically permissible. Optimizer resets and a small number of steps can affect learning, but the available evidence does not establish that either explains the performance ceiling.

### Implementation tasks

1. Preserve AdamW state across completed RL iterations and through preemption. Record the first checkpoint from which moment continuity is available.
2. Keep each root's admitted samples together in an optimizer chunk for the repaired variant. Calculate advantages from the complete sampled group before chunking.
3. Preserve separate policy references: frozen S0 for KL, and the appropriate current rollout policy and clipping baseline for the update. Persisting optimizer state must not reuse an earlier iteration's clipping baseline.
4. Align comparison configurations. The checked-in main arm has a row-gradient bound of 1, while the sampled comparison still has 25; these cannot support a clean objective comparison. [Recorded configuration change](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/research/hpc/compound_rl_20261107/README.md#L150-L160)
5. Report policy, KL, and auxiliary gradient contributions; clipping fractions; dominant rows; lifecycle action probabilities; and effective optimizer steps.

### Required verification

Use a small deterministic model to check continuous training against save-and-resume training across both step and iteration boundaries. Every admitted row must be accounted for exactly once per declared pass, with no duplicate updates or leftover accumulated gradients.

Check that equivalent normalization gives the same aggregate objective and gradient before the optimizer is applied. Sequential Adam steps are not expected to equal one combined step. Include an adversarial root with contrasting successful and failed samples, invalid sampled outputs, and a preemption during an update.

Start calibration with two whole-root chunks and one pass. If decision movement remains weak, compare four whole-root chunks on one fixed training-only rollout batch. Select the smallest stable schedule that improves intended decisions while retaining correct commits. Preserve the approved row bound initially; changing chunk normalization changes its practical effect, so inspect the resulting clipping distribution.

Use a fixed decision panel containing true first candidates, true later candidates, wrong accepted candidates, and mixed breaker-plus-parameter cases. A falling loss or increased decline probability alone is insufficient.

**Exit gate:** reproducible finite updates, correct likelihood provenance, passing resume checks, matched configurations, and useful action-level changes on the panel. Failure here redirects effort to the updater before any conclusion about history or RL is drawn.

## Stage 1 Restore complete observable tool history

Introduce an explicit observation mode that retains every eligible tool result already emitted during the episode. Apply the same renderer during rollout, update, greedy evaluation, and resume.

The current canonical observation keeps the last four history events; downstream compaction also removes older WLS tables and applies a character budget. Changing only the adapter's history setting would therefore miss the upstream restriction. [Canonical observation](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/dagger/evaluator.py#L4326-L4378), [History compaction](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/dagger/dataset_builder.py#L1240-L1342)

### Retain evidence with its state

Each event should preserve its action, arguments, returned observable results, and temporal order. Use stable canonical aliases to associate it with the active state, an open candidate, or a previous candidate experiment. Track whether the candidate was committed or discarded.

Keep the current active-state summary easy to locate, followed by the relevant candidate evidence and the complete historical record. After rollback, the restored parent's evidence is current; the declined candidate remains a historical comparison. After commit, old evidence remains readable but cannot authorize actions against the new state.

Build the record from the sanitized policy transition stream. Offline truth, compound-class labels, target-membership labels, future outcomes, and evaluator assessments remain outside the actor observation. The evaluator already distinguishes observable history from the richer audit trace. [History and trace construction](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/dagger/evaluator.py#L2772-L2799)

Complete history restores previously returned information. The WLS provider itself emits bounded tables, so recovering rows never returned by the tool would be a separate evidence intervention. [WLS table generation](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/providers/wls_tables.py#L35-L133)

### Acceptance checks

- An early WLS table and a candidate edit remain visible after more than four subsequent actions.
- Decline preserves historical evidence and correctly restores the parent-current evidence.
- Commit updates permissions and state aliases without erasing earlier observations.
- Changing only hidden audit metadata leaves the policy prompt unchanged.
- The existing observation mode reproduces its previous prompts.
- History is never silently removed through event limits, table removal, character budgets, depth limits, or tokenizer truncation.

Render existing trajectories before training. Report median, 95th-percentile, and maximum prompt tokens by compound class and action position, as well as total input tokens per episode. Check the configured 32,768-token training limit and reserve room for completions. [Current training configuration](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/research/hpc/compound_rl_20261107/rl_ieee14.json)

If complete histories exceed the usable limit, first remove redundant serialization without losing information. If indexed retrieval is needed, define and evaluate it as a separate history-access mode; include its costs and preserve every difficult root in the evaluation.

**Exit gate:** history fidelity, state binding, and token-budget checks pass. A frozen model does not have to improve for the implementation to pass; improvement is the next experiment.

## Stage 2 Diagnose recoverable decisions

Use the same saved learner-reached states to test both observation modes and the consequences of each legal lifecycle choice. The aim is to find where better evidence or better decisions can change completed outcomes.

### Diagnostic sample

Start with 32 distinct training roots and 16 additional roots reserved for diagnostic validation. Select one accepted-candidate fork per root where commit and decline are both legal. Freeze the sampling rules and record any coverage shortfalls.

Cover wrong accepted candidates, true first candidates, true candidates after an earlier decline, breaker-only false parameter candidates, and mixed breaker-plus-parameter true candidates. Include roots the expert fails and successful roots that test retention. Related parent scenarios and their noise variants stay in the same split.

First compare the frozen checkpoint's decisions under the current and complete-history observations: 48 states times two views gives 96 decision evaluations.

Then run both legal choices, under both history modes, with two learner continuations per choice:

**48 states × 2 choices × 2 observation modes × 2 continuations = 384 suffix episodes.**

This count excludes prefix replay and diagnostic evaluation overhead. Reuse the original remaining budget, including every action already spent in the prefix. Where estimates are unstable, a predeclared expansion to four continuations per choice raises the maximum to 768 suffix episodes.

The simulator state, controller state, candidate-parent relation, counters, and visible history must match at each fork. Seed schedules should match where practical. Record the forced fork action as an intervention, separately from subsequent sampled learner actions.

### Breaker only and mixed cases

The latest reported candidate audit is preliminary:

| Audit slice | True line-parameter candidates |
| --- | ---: |
| Breaker-only classes | 0 of 111 |
| Other classes | 1,990 of 2,086 |
| Mixed breaker and line-parameter cases | 567 of 608 |

The mixed cases are a subset of the other-classes row. These are conditional candidate counts, not independent episode success rates. Reproduce the counts from source records before using them to set sampling weights.

The decision the learner needs is whether breaker errors alone explain the apparent parameter anomaly. Preserve true parameter repairs when breaker and parameter faults coexist. Test the visible evidence before and after topology correction, the candidate's numerical effects, remaining discrepancies, and the next legal hypothesis. Scenario-class labels are used only for offline stratification.

### Record the complete recovery chain

For each fork, record whether a wrong target was declined, whether a productive next target was available and chosen, whether the numerical correction worked, and whether the episode ultimately solved. Record wrong commits and false declines separately from final success.

Use both root-weighted and decision-weighted summaries. Conditional rates from complete episodes can differ because policies encounter different candidates; the fixed-state fork experiment is the cleaner comparison of decision consequences.

**Proposed screening gate:** demonstrate successful learner continuations on at least five distinct expert-failed roots, with successful recovery also present in the diagnostic-validation split, before funding a continuation-focused training arm. This is a compute-allocation rule, not a significance threshold.

If memory improves choices but recoveries still fail, focus on subsequent routing and continuation credit. If the frozen checkpoint does not improve, retain the possibility of a short trained history pilot: the representation has changed from its training format. Zero sampled recoveries indicates weak observed continuation support; it does not prove that recovery is impossible.

## Stages 3 and 4 Run matched RL pilots

Use one starting adapter, the same repaired optimizer schedule, the same root schedules and seed plans, the same legal actions, the same numerical tools and verifier, and the same episode budget. Keep reward definitions fixed for the first history comparison.

| Arm | Observation | Learning objective |
| --- | --- | --- |
| A | Current bounded history | Current episode and lifecycle objective |
| B | Complete observable history | Same objective as A |
| C | Current bounded history | Decision and continuation credit |
| D | Complete observable history | Same objective as C |

The current objective already includes supervised lifecycle targets alongside policy-gradient and KL terms. Its lifecycle label identifies whether the requested target belongs to the injected fault set; it does not by itself establish that all numerical edits were helpful or that committing improves the completed outcome. [Lifecycle labels](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/rl/lifecycle.py#L67-L145), [Update losses](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/rl/update.py#L718-L789)

### First comparison

Run A and B first, after a small end-to-end smoke. Budget two completed iterations per arm at 64 roots and four trajectories per root: **512 sampled full episodes per arm, 1,024 across A and B**, plus baseline and checkpoint evaluations.

Evaluate S0 under each observation mode. This separates the immediate effect of changing the input from learning under that input. Regenerate each arm's rollouts with its actual renderer and behavior policy; changing prompts in an old dataset does not create on-policy experience.

Measure development results after each completed iteration. The first screen asks whether full history produces new solves, preserves correct commits, and justifies its added input-token and GPU cost.

### Continuation credit

Prepare C and D only when the fork audit supports an actionable recovery signal. Keep A and B as the matched controls and start C and D from the same original checkpoint for the factorial comparison.

For critical sampled decisions, estimate the completed return of the legal alternatives using learner continuations from the same state. The proposed first method uses a fork-local advantage for the actually sampled action, with an action-independent baseline estimated from separate branch samples. Use this local advantage on decisions with continuation targets and the existing episode advantage on other sampled tool actions. Replace membership-only auxiliary supervision on those covered decision rows; retain the documented objective elsewhere. Keep the frozen-S0 KL anchor. Record this whole decision-credit change as the C/D treatment.

The first implementation should concentrate outcome credit on commit and decline. Add the next diagnostic choice after a decline where the probe establishes a tractable legal branch comparison. Preserve terminal audited success and healthy-component preservation as the governing outcomes. Audit whether current target-membership supervision conflicts with realized repair quality, including every line changed by a pair refit.

Forced alternatives may provide action-value targets. They are not ordinary sampled actor rows. Policy-gradient rows must retain the actual behavior-policy likelihoods and the appropriate on-policy or explicitly corrected sampling contract.

Begin with Monte Carlo continuation estimates before adding a learned critic. This adapts the credit-assignment idea studied in VinePPO; its published results concern mathematical reasoning, so the power-system setting requires its own validation. [VinePPO, ICML 2025](https://proceedings.mlr.press/v267/kazemnejad25a.html)

Enrich training for informative forks while preserving the natural-distribution validation set. Report enrichment weights and keep easy successful roots in the retention panel. A large share of equal-return groups is a signal to inspect exploration and recovery support. Preserve useful auxiliary or retention data from those groups even if their group-relative policy advantage is zero.

Use the same two-iteration screen for C and D: another 1,024 sampled full episodes. Initially cap continuation augmentation at 32 fresh decision states per arm per iteration, with two choices and two continuations per choice: up to 128 suffix episodes per iteration, or 256 per arm across the pilot. Regenerate values for the current policy and observation mode; earlier checkpoint values are diagnostic records, not current on-policy targets. Report this continuation cost separately.

### Interpret the comparison

- B improving over A supports a useful contribution from retained history.
- C improving over A supports a contribution from decision and continuation credit.
- D improving when B and C do not suggests the two changes work together.
- Better local rejection with unchanged completed success identifies a remaining recovery problem.

If gains are attributed specifically to outcome RL, add a gated policy-gradient-on versus policy-gradient-off control with the same auxiliary supervision and matched learner-state data. The existing hybrid objective otherwise supports a claim about the combined method. This diagnostic reuses the learner's experience and does not require a new expert DAgger round.

## Conditional work on actions and numerical evidence

Open this work only when replay identifies a specific blocker.

### Candidate availability

Audit whether useful targets can become legally available along any practical current route. A missing target in the first menu alone is insufficient evidence that the action space blocks recovery.

The current ambiguous parameter menu is restricted to two distinct lines, and unsupported correction targets are rejected by the controller. [Candidate menu](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/providers/matpower.py#L3175-L3203), [Target support checks](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/oracle/process_validity.py#L874-L901)

If useful targets remain inaccessible, test one bounded expansion mechanism, such as requesting the next observable candidate batch. Bind it to the active state, keep correction verification unchanged, and charge its actual tool and numerical cost. Verify that hidden truth cannot change the menu.

### Additional evidence

If retained outputs cannot discriminate useful decisions but full numerical records can, expose the smallest additional operator-observable record that resolves the ambiguity. Candidates include signed residual and multiplier changes, identities of remaining adjacent anomalies, and the actual before-and-after edits for every line touched by pair refitting.

The existing detailed evidence already computes some neighborhood summaries before table caps, so new fields should add information beyond those summaries. [Detailed numerical evidence](https://github.com/yifeihsu/psse_agent/blob/4b14d5ed4f8495d07ae5bc6583e88fde72bde56c/psse_env/providers/detailed_evidence.py#L160-L198)

Keep these interventions distinct from the history pilot. Report capability changes explicitly and compare policies under the same interface wherever supported, while preserving the original fixed rule expert as the project reference. Improving the expert's decision rules is outside this workplan.

## Evaluation and stopping rules

### Scorecard

| Metric | Purpose |
| --- | --- |
| Audited and clean solves on all roots | Main outcome and preservation check |
| Learner-only solves and reference-only solves | Paired gains and losses against S0, the corrected control, and the expert |
| Expert-failure roots recovered | Evidence that the learner adds useful capability |
| False declines of true candidates | Retention, split by first and later candidate and by mixed-error slice |
| Wrong targets committed and healthy-component corruption | Whether gains trade away correct system state |
| Correct declines followed by a useful next correction and solve | Recovery mechanism |
| Invalid actions, loops, and false finalizations | Process failures |
| Tool calls, internal WLS/refit work, input/output tokens, and runtime | Actual inference and training cost |

The primary paired improvement is learner-only solves minus reference-only solves. Report both counts, not only the net.

Use the repeatedly inspected 300-case set for development selection. A score above 188 there is a useful milestone, but does not establish general superiority. Evaluate efficiency on the intersection of roots solved by both policies, alongside overall success and failure costs.

If testing tighter budgets, run separate evaluations in which the policy actually operates under 16 or 20 actions. A case finishing within 16 actions during a 40-action run remains a descriptive within-run measure.

### Pilot promotion

Promote a pilot only when it has positive paired net solves over its matched reference, actual recovery of expert-failed roots, and nonnegative paired clean-success change. Use S0 as the reference for the repaired control and the corresponding corrected arm for an observation or credit intervention. Require no increase in healthy-component corruption on the fixed development panel. Inspect true-candidate false declines, especially mixed and later-candidate cases, before increasing its budget.

These small-sample gates allocate compute; they do not establish statistical significance or eliminate uncertainty. A local classification gain without new completed recoveries is a diagnostic result.

Keep the current run's approved stopping rule attached to that run. For the new series, predeclare a two-iteration screen and permit at most one further iteration to resolve a concrete uncertain signal. Stop expansion after three completed valid iterations without end-to-end improvement over the arm's own initial observation-mode baseline. Local decline accuracy does not reset this counter.

Stop immediately on nonfinite updates, incorrect likelihood provenance, silent history truncation, duplicated or missing update work, or invalid checkpoint recovery. A preemption is neither a completed iteration nor an algorithmic failure.

### Confirmation

Freeze the selected method, observation schema, reward, budget, and checkpoint-selection rule. Compare the selected method and corrected control across three training seeds; pilot seeds may count if their settings were already final. Report each seed separately.

Choose an untouched test set and its size before opening results, using pilot effect estimates and the expected rate of discordant pairs. Keep related scenario roots and noise variants together. Report paired confidence intervals and an exact paired test where appropriate. Repeated evaluations on the same roots are not independent extra test cases.

The expert-beating claim requires a positive paired effect on that untouched test with uncertainty narrow enough to support the claim, together with the clean-success and preservation results. Inconclusive results remain inconclusive; do not extend the test repeatedly until a significance threshold is crossed.

Evaluate IEEE 118 transfer only after the design is frozen, under the declared supported error families. Keep the earlier measurement/line-parameter transfer result separately identified; it does not establish breaker-status transfer.

## Implementation tickets and execution order

| Ticket | Main code area | Reviewable output |
| --- | --- | --- |
| T1 Updater continuity | `psse_env/rl/update.py`, `train.py`, update tests | Root chunking, optimizer persistence, resume tests, component diagnostics |
| T2 Matched run manifests | `research/hpc/compound_rl_20261107/`, config tests | Pinned configurations, source hashes, baseline register, stopping rule |
| T3 Complete history mode | `dagger/evaluator.py`, `dataset_builder.py`, policy adapter and factory | Observation contract, state-version tests, token audit |
| T4 Fork probe | New focused research harness reusing compound replay code | Frozen state manifest, branch traces, paired continuation report |
| T5 Decision credit | `rl/lifecycle.py`, `rows.py`, `targets.py`, `update.py` | One specified objective and behavior-provenance checks |
| T6 Evaluation report | RL evaluation and research reporting | Paired scorecard, recovery chain, strata, costs, seed summary |

The first work block completes T1 and T2 while T3 and the T4 harness are prepared in parallel. The next block runs the history fidelity checks, updater calibration, and fork probe. The first available training allocation then runs A and B. C and D follow only when their recovery and implementation gates pass. T6 accompanies every experiment.

Budget scientific progress by completed valid updates, sampled trajectories, continuation episodes, and active GPU time. Track queue delay and preemption overhead separately. Measure the changed prompt lengths and repaired update schedule in the smoke before committing to a wall-clock forecast.

The first review should have four concrete outputs: a verified updater, a complete-history token and fidelity report, a paired recovery audit, and fully specified A/B configurations. Those outputs make the next training expenditure reviewable and identify which mechanism the run is testing.

