# Hard cases and reinforcement learning for the diagnostic agent: assessment and plan (2026-10-13)

Branch `codex/classifier-triage-20261004`. Answers the question of 2026-10-12
(the rule-based expert's 159 of 160 looks unrealistically optimistic; build
edge cases where the rules cannot decide or only rank candidates and hand
off; use reinforcement learning there for accuracy and to cut unnecessary
tool calls). Read against the code at `ffb9dd2`, the expert end-to-end runs
under `output/classifier_triage_20261004/e2e/` and
`output/classifier_triage_20261013/e2e_*`, and the cell records of
2026-10-07, 2026-10-11 and 2026-10-13. Assessment and plan only; nothing here
has been built.

Implementation (2026-10-14): steps 1 to 4 are built on this branch; see
`docs/rl_experiments_20261014.md` for what exists, how to run the pilot and
what is still open.

## 1. Decision in one paragraph

Yes to both goals, but the measurements point RL somewhere other than the
question assumes. The 159 of 160 is a property of how the suites are drawn:
four admission filters keep only roots the teacher solves on its first pass,
and the one development root outside that envelope is the one it fails. On
the development draw the current teacher does almost no trial and error (six
rollbacks in 160 episodes, all on that root), so an efficiency optimizer has
about 3% of calls to recover there under the present contract. The search
cost the question describes lives in two other places: the student once it
leaves its training distribution (IEEE 118: 20 false commits and 1.3 extra
calls per root against the expert), and a per-hypothesis protocol cost that
is a contract choice, not a learned behaviour. So: build the hard draw first
and publish the expert's ceiling on it; decide a reward that ranks a decisive
diagnosis above a ranked-list handoff; add sampling and a group-relative
policy-gradient step on top of the existing collector, audit and LoRA
adapter; and run RL against a DAgger round on the same hard roots as the fair
control. Three cases are unlearnable in the present action space and must be
labelled as such before any learning is scored (section 5).

## 2. Why 159 of 160 is the ceiling of the draw, not of the method

### 2.1 The admission filters

| Filter | Where | Effect |
| --- | --- | --- |
| Admission mode `recoverable` (the default) | `psse_env/examples/generate_round0_aggregate.py`, `--admission-mode` help: "recoverable admits teacher-solvable roots (the IEEE 14 contract)" | A root enters D0 or a suite only if the teacher's own residual test and route solve it |
| Parameter dominance threshold 1.2 for training and collection roots | `research/hpc/full_pipeline_20260907/pipeline.env`, `SUITE_TRAINING_THRESHOLD` ("teacher-solvable on the first pass") | Training roots are those where the true line dominates the Lagrange-multiplier ranking |
| Development threshold 1.0 with rank allowance 2 | `pipeline.env`, `SUITE_DEVELOPMENT_RANK_ALLOWANCE`; `psse_env/providers/scenario_generator.py:1399` raises `ScenarioRejected("parameter_context_target_ambiguous")` | A parameter root whose true line ranks third or lower in the deployed context never reaches the development set |
| Meter errors lifted to at least 10 noise sigmas | `pipeline.env`, `MEASUREMENT_ERROR_MIN_SIGMA=10` | No weak meter error in D0 or the suites |
| Detectable-only waveform corpora | the HIF and unbalance pools admit only windows that alarm the balanced WLS (118 of 420 HIF and 160 of 440 unbalance windows at the 2026-09-21 regeneration) | No sub-threshold HIF or unbalance |

### 2.2 What that leaves in the development draw

Counted from `output/classifier_triage_20261004/suites/development.json`
(160 roots, 12 families): of the 32 parameter-bearing roots, 29 are
`dominant`, 2 `ambiguous` and 1 `misranked`. In the unfiltered population of
the step-1 identifiability study (250 parameter roots,
`docs/hypothesis_ranking_step1_20260930.md`) the split is 194 dominant, 48
ambiguous and 8 misranked. The context tools pre-rank the candidates, so on
most roots the agent's decision is "take the first candidate": the first
correction hits the true target on 31 of 32 parameter roots and 64 of 64
meter roots (`docs/hypothesis_ranking_step3_20261001.md`).

The single failure, `r0_4d7aa8a5a01f`, is the misranked root: the wrong line
2-3 passes observable verification and is committed, the true line 3-4 is
adjacent, and the baseline expert then spends 38 calls and 6 rollbacks before
the budget handoff. The ledger teacher names the right line through the
screen's refits and closes it in 6 calls; the baseline teacher's parameter
context never offers the second candidate. Under the deployment verifier
both corrections fit the measurements, so this is an identifiability ceiling
of the verifier, not an ordering defect.

### 2.3 Hard cases that already exist outside the draw

- Same-sign flow-meter pairs on one line mimic an HIF: the balanced screen
  flags 37.5% of them at its HIF recall (`docs/hypothesis_ranking_step4_20261001.md`);
  the ledger teachers solve 17 of 24 opposite-sign mimics against 21 for the
  baseline (`docs/hypothesis_ranking_plan_20260930.md`, section 12).
- IEEE 118: the expert solves 119 of 124 development roots (two HIF, two
  HIF-plus-meter and one parameter root missed; `docs/triage_family_scores_20261013.md`).
- At matched detector effect (band 1.5 to 20, `docs/effect_targeted_magnitude_20261012.md`)
  the triage gate's zero-shot recall from IEEE 14 to 118 is 66.6%, with
  harmonic 12 of 55 and unbalance 14 of 83: every gate miss on a waveform
  root becomes an agent root that enters the balanced ladder without the
  evidence it needs.

## 3. What the efficiency data say

### 3.1 The teacher on the 160 development roots

Per-root tool calls of the teachers after the 2026-10-07 contract
(`output/classifier_triage_20261004/e2e/*_v2`) and after the 2026-10-13
spectra-first route (`output/classifier_triage_20261013/e2e_ieee14_spectra_first`):

| Teacher | Success | Calls | Mean / median per root | Rollbacks | Corrections beyond commits | Phasors / spectra fetched |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier profile, baseline expert (2026-10-07 contract) | 159 | 1,161 | 7.26 / 6 | 6 (one root) | 6 | 59 / 18 |
| Screen profile, ledger + ranker (2026-10-07 contract) | 160 | 1,137 | 7.11 / 6 | 0 | 0 | 58 / 18 |
| Classifier profile, spectra-first route (2026-10-13) | 159 | 1,129 | 7.06 / 6 | 6 (one root) | 6 | 43 / 18 |

Mean calls by family for the current teacher: healthy and telemetry-only 2,
unbalance 4, harmonic 4, HIF 5, single meter, parameter or topology 6,
HIF with a meter 9.1, meter plus parameter or topology 10, multi-meter 17.6,
parameter 8 (6 without the misranked root).

The floor under the current contract for a single-fault root is six calls:
WLS, context, correction, verification WLS, commit, confirmation handoff.
The teacher is at that floor on eleven of twelve families. Summed over the
draw the floor is about 1,090 calls against 1,129 spent: the recoverable
slack is about 3%, and 32 of those 38 calls are the misranked root. The
"roll back and try the next candidate" loop fired on one root.

### 3.2 Where the search cost really is

1. **The student off-distribution.** On the 124 IEEE 118 development roots
   (`docs/triage_family_scores_20261013.md`, cell A):

   | Policy | Success | False commits | Episodes with invalid actions | Mean steps |
   | --- | --- | --- | --- | --- |
   | Frozen IEEE 14 R2 student, 2026-10-11 environment | 101 | 26 (12 on harmonic roots) | 46 | 10.0 |
   | Frozen IEEE 14 R2 student, spectra-first environment | 115 | 20 | 30 | 8.3 |
   | Adapted after one IEEE 118 DAgger round | 114 | 7 | 21 | 7.6 |
   | Expert | 119 | 0 | 4 | 7.0 |

   The 2026-10-11 run also had 98 invalid actions, 16 rollbacks and 24 loop
   episodes. The harmonic gap (0 of 16) closed from the environment change
   alone, with nothing retrained. What remains is the student's regret: 1.3
   calls per root above the expert, 20 false commits, loops on the roots the
   routing misroutes. On IEEE 14 the adapted adapter spends 7.3 calls per
   root against the expert's 7.0 and makes 8 false commits (cell B,
   156 of 160).
2. **The per-hypothesis protocol.** Multi-meter roots cost about four calls
   per meter (context, correction, verification, commit) because the teacher
   repairs one meter per pass. The correction argument `suspect_group`
   already accepts a list of channels and the measurement expert has a
   joint-retry path, so batching is a teacher and contract decision. Batched
   repairs would take the twelve multi-meter roots from 211 calls to about
   85, about 11% of all calls on the draw, more than any learned policy can
   save elsewhere.
3. **Mandated steps.** Verification and the confirmation handoff are
   contract requirements (`docs/classifier_triage_plan_20261004.md`,
   section 12). No policy goes below six calls on a single fault unless the
   contract changes.

## 4. Where RL fits

RL is the right tool for three decisions the rule set makes by fixed order:
which hypothesis to verify first when the ranking is close, whether an
acquisition is worth its cost on this state, and when to stop verifying and
hand off a ranked list. Most of the machinery exists.

- **A verifiable terminal reward.** `psse_env/dagger/release_audit.py`
  scores truth-audited success by basis (`counterfactual_resolution`,
  `bounded_localization_handoff`, `unexplained_discrepancy_handoff`), and
  the evaluator records false commits, false finalizations, healthy
  components touched, invalid actions, loops, steps, WLS calls and
  specialized tool calls per episode. `psse_env/dagger/evaluator.py:137`
  already holds a score-weight dict (success +4, false finalization -5,
  healthy-component corruption -5, false commit -4, false rollback -2,
  loop -2, forced-error recovery +2, partial retention +2, tool regret
  -0.25). The tool-regret term has never been populated (coverage 0 in every
  summary); it is the natural home for an efficiency reward measured as
  calls minus the expert's calls on the same root.
- **A cheap environment, an expensive policy.** The expert rolls out 160
  roots in 4.1 min on one CPU, about 0.2 s per call. The 12B student costs
  about 1.5 s per step on an H200 (124-root zero-shot evaluation in 30 min;
  cell B's 160 roots for two policies in 43 min). Eight sampled trajectories
  per root over the 122 round roots is about 1,000 episodes and 8,000
  generations: two to four hours per iteration on one 80 GB GPU with the
  existing two workers, so five to eight iterations a day inside the current
  cell shape.
- **A fitting algorithm.** Group-relative advantages per root (GRPO or
  RLOO) need no critic and match the root-grouped suites: sample K
  trajectories from one root, advantage equals the trajectory's reward
  minus the root's mean. Broadcast the terminal reward to every assistant
  turn of the trajectory; the per-call cost inside the reward gives the step
  penalty without separate credit assignment. Loss on assistant tokens only,
  which the SFT gates already mask (`completion_mask` in
  `psse_env/sft/gates.py`); LoRA parameters only; a KL penalty to the R2
  adapter as the reference. TRL 0.24 is pinned and ships `GRPOTrainer`, but
  it generates single-turn completions; the multi-step tool loop needs the
  collector as the rollout engine and a hand-written per-step token loss,
  or a flattening of each step into a (prompt, completion, advantage)
  sample. Two code changes are required before any of this:
  `psse_env/dagger/research_policy_factory.py:594` decodes greedily
  (`do_sample=False`, temperature 0) and needs a sampling mode with K
  samples per observation and stored per-token log-probabilities; the audit
  needs a graded episode score beside the boolean.
- **Dense shaping is available but must stay small.** The collector's
  `all_admissible` supervision already exposes the expert's admissible
  proposal set at every visited state. A bonus for choosing an admissible
  action is a legitimate process term, but at more than a tenth of the
  success credit it re-imposes the teacher's ceiling on exactly the roots
  the hard draw exists for.

## 5. Design conflicts to decide before building

Stated plainly, because each one changes what gets built.

**C1. What "better than a handoff" means.** Under the present audit a bounded
handoff naming two lines with the truth among them is full success. A flat
reward teaches the policy to hand off. Recommended ordering: decisive
correct diagnosis above a bounded handoff containing the truth, above an
unexplained-discrepancy handoff, above budget exhaustion; false commits and
healthy-component edits below all of them; then a per-call cost and a cost
per unneeded acquisition. Let the ranked-list handoff carry a policy-chosen
list length k with credit decreasing in k (for example 1.0, 0.6, 0.4 for
k = 1, 2, 3), so ambiguity becomes a priced action rather than a terminal
fallback.

**C2. Discriminating evidence for the identifiability-limited roots.** On
the misranked root both corrections pass observable verification; on
balanced SCADA an unbalance is indistinguishable from a bad voltage meter;
a same-sign flow-meter pair is indistinguishable from an HIF until the
phasors arrive. RL can learn to request the discriminating evidence only if
such an action exists: a second snapshot or operating point, branch-current
telemetry, or the deferred close-alternative protocol that verifies both
candidates on the same parent and compares the untargeted consistency gain.
Without one of these, the correct learned behaviour is a two-candidate
handoff and the hard set must label it as the target. Decision: which, if
any, of these actions enters the action space for the hard draw.

**C3. Which contract costs are negotiable.** The six-call floor, the
one-meter-per-pass repair and the context fetch before a correction are
contract choices. Decide before measuring efficiency whether batched meter
repairs and a direct correction from the WLS table (no context fetch when the
table already names the target) are allowed. If they are, the expert gets
them first and the RL arm is measured against that expert, so contract
savings are never reported as learning.

**C4. Decoding.** RL needs sampling at a temperature above zero and K
rollouts per root during collection; evaluation stays greedy so the numbers
remain comparable with every earlier cell. Invalid actions under sampling
are penalized by the existing schema gate and count against the budget as
today.

**C5. The control arm.** A DAgger round on the same hard roots is the fair
baseline, not the frozen R2: more imitation on hard roots may close much of
the gap by itself (on IEEE 118 one round changed habits but not the count).
RL is credited only for what it adds over that arm, paired root for root.

**C6. Drawing hard roots in the pipeline.** `Round0ScenarioGenerator`
accepts `effect_band=` and `parameter_target_rank_allowance=`, but
`generate_round0_aggregate.py`, `build_suite.py` and `pipeline.env` expose
neither the effect band nor an admission mode for the stratified hard draw.
That plumbing is step 1's first task.

## 6. Experiments

### Step 1 (CPU, one to two days): the hard draw and the expert ceilings

1. Expose the effect band and a `hard` admission mode to the round-0
   generator and the suite builder; record stratum, effect ratio and the
   identifiability tag (step 1.3) on every root under `audit`.
2. Draw a hard development set and a hard training set with the same
   stratum plan. Proposed development composition (about 300 roots; the
   training draw doubles it):

   | Stratum | Why it is hard | How to draw it | n |
   | --- | --- | --- | --- |
   | Parameter, misranked | a neighbour outranks the true line; the wrong fix passes verification | admission `physical`, keep `true_line_rank >= 2` (8 of 250 in the population, so generate about 1,000 parameter roots or construct adjacent-line pairs) | 24 |
   | Parameter, ambiguous | dominance ratio 1.0 to 1.2 | keep roots the training threshold rejects | 24 |
   | Meter plus parameter on the same or adjacent line | both hypotheses explain the residual | generator overlay keyed to the faulted line | 16 |
   | Weak single meter | effect ratio 1.5 to 3 instead of at least 10 sigma | effect band on the balanced corpus builder | 24 |
   | Multi-meter with a same-line pair | the pair mimics an HIF (same-sign and opposite-sign) | the mimic builder of `research/classifier_triage/ieee118_dataset.py` | 24 |
   | HIF at the band edge | 200 to 500 and 500 to 1000 ohm, effect 1.5 to 3 | the physical corpora's upper bands | 24 |
   | HIF with a meter | the expert already misses two of eight on IEEE 118 | as in the pipeline, low effect | 16 |
   | Unbalance and harmonic at low effect | the gate and the routing are weakest here | effect band 1.5 to 3 | 16 + 16 |
   | Two parameter errors on adjacent lines | identifiability | new generator overlay | 16 |
   | Meter plus topology, parameter plus topology | mixed families with a breaker | existing overlays | 16 + 8 |
   | Alarmed healthy windows | false alarms (40 in `ieee14_effect`) | balanced corpus healthy rows that alarm | 16 |
   | IEEE 118 subset across the same strata | network transfer | `ieee118_suite.py` with the band | 48 |

3. Tag every root offline with what the deployment verifier can separate:
   `separable_by_verification` (only the true correction passes),
   `separable_by_acquisition` (phasors, spectra or branch currents decide),
   `not_separable` (the top wrong candidate also passes and no admitted
   stream discriminates). The counterfactual generator
   (`psse_env/dagger/counterfactual_generator.py`) already executes
   isolated branches per root and the candidate oracle runs in deployment
   mode, so this is a scripted pass. The separable fraction is the
   attainable ceiling for any policy; report it per stratum.
4. Run three expert arms on both hard sets under the deployment verifier:
   the baseline expert on the classifier profile, the ledger expert with the
   ranker on the screen profile, and the spectra-first teacher. Report
   success by basis and by stratum, calls per root, acquisitions and
   unneeded acquisitions, rollbacks, false commits, healthy components
   touched.

Decision rule: a hard development set on which the best expert arm scores
above 95% is not hard enough; one below the separable fraction minus 10
points means the draw is dominated by unlearnable roots and the strata are
rebalanced. Target 60 to 85% for the best arm, with the `not_separable`
share reported beside it.

### Step 2 (one day): the reward contract

1. Add a graded episode score to the audit: basis credit under C1, minus
   lambda per call, minus mu per unneeded acquisition, with the existing
   penalties for false commits, false finalizations, healthy edits and
   loops. Keep the boolean success by basis unchanged and report both.
2. Populate `tool_regret` with calls minus the expert's calls on the same
   root (the expert reference comes from step 1), so efficiency is read as
   regret against a known-good trajectory and not as raw length.
3. Extend the bounded handoff to a policy-chosen list length up to K, credited
   when the truth is in the list and discounted by length; leave the
   provider's two-line allowance as the default so earlier cells still audit
   the same way.
4. Predeclare lambda, mu and the discount schedule from the step-1 expert
   arms (for example lambda such that one extra call costs 2% of a decisive
   success), and do not tune them on the hard development set.

### Step 3 (two to three days): sampling, rollouts and the update

1. Sampling mode in the policy factory: temperature and top-p, K samples per
   observation, per-token log-probabilities of the chosen completion
   returned with the action; greedy stays the evaluation default.
2. A trajectory store written by the collector: per step the rendered prompt,
   the completion tokens, the log-probability sum, validity, the step index;
   per episode the audit record and the graded score. K independent episodes
   per root, differing only in the policy's samples.
3. The update: group-relative advantage per root (RLOO leave-one-out mean or
   GRPO normalization), token-level loss on assistant tokens through the SFT
   collator's masks, LoRA parameters only, KL penalty to the R2 adapter,
   PPO-style clipping only if samples are reused across more than one
   update. Start from the R2 adapter, not the base model.
4. One iteration: sample K per root over the hard training set, score,
   update; greedy evaluation on the standard 160 and the hard development
   set after every iteration (about an hour). Smoke test on three roots on
   the cluster; the local machine cannot serve the 12B model at this length.

### Step 4 (HPC cell, about a week wall clock): the paired comparison

Three arms, paired on the hard development set and on the standard 160, with
IEEE 118 as a zero-shot transfer check for each:

| Arm | Policy | What it isolates |
| --- | --- | --- |
| 1 | R2 frozen (the 2026-10-07 classifier cell's adapter) | The gap imitation leaves on hard roots |
| 2 | DAgger round 3 on the hard training set, same episode budget as arm 3 | What more imitation buys |
| 3 | RL from R2 on the hard training set | What the truth reward adds beyond imitation |

Primary outcomes: truth-audited success by basis and by stratum, calls per
root on roots both arms solve, unneeded acquisitions, false commits, healthy
components touched, loops; parent-bootstrap intervals over physical roots.
Deployment criterion for the RL arm: it beats arm 2 on hard-set success by a
margin whose interval excludes zero at equal or fewer calls, and it does not
fall below 158 on the standard 160. Report efficiency separately for the
contract changes of C3 (expert with and without batching) so the two sources
of savings stay distinguishable.

## 7. What not to do

- Do not give the policy any truth-derived signal during the episode; the
  reward is computed by the offline audit at the end, as today.
- Do not reward agreement with the expert's action on hard roots beyond a
  small process term; the hard draw exists because the expert is wrong or
  undecided there.
- Do not let the per-call cost dominate the success credit; the cheapest
  policy is one that hands off at the first alarm.
- Do not credit a handoff without the truth-in-list test, and do not merge
  the bases into one success number.
- Do not tune lambda, mu, the list discount or the KL weight on the hard
  development set; fix them from the step-1 expert runs.
- Do not report savings from batched repairs or a skipped context fetch as
  learning; run the expert with the same contract first.
- Do not sample the evaluation; greedy decoding keeps every earlier cell
  comparable.

## 8. Order and effort

| Step | Depends on | Effort | Output |
| --- | --- | --- | --- |
| 1 Hard draw, identifiability tags, expert ceilings | C2, C3, C6 | 1 to 2 days CPU | `output/rl_hard_cases_20261013/` with both draws, strata, tags and the three expert arms |
| 2 Reward contract, graded audit, tool regret, k-list handoff | C1, step 1 numbers | 1 day | audit and evaluator changes with tests |
| 3 Sampling, trajectory store, policy-gradient update | step 2 | 2 to 3 days plus a cluster smoke test | `psse_env/rl/` (or an extension of the collector) with a three-root receipt |
| 4 Paired cell | steps 1 to 3 | one cell, about a week | three-arm report by basis and stratum, IEEE 118 transfer |

## 9. Corrections to the question's premises

- The fallback loop is not where the expert's success comes from on the
  development draw: the ledger teacher makes no rollback at all and the
  baseline teacher's six are on one root. The loop is the student's, off
  distribution.
- A ranked-list handoff is not a new outcome: `operator_escalation:ambiguous_branch_candidates`
  with a two-line allowance exists and is credited by the audit when the
  truth is among the lines (`psse_env/test_ambiguity_handoff.py`). What is
  missing is a reward that prefers resolving it.
- The clearest student failure of 2026-10-11 (harmonic 0 of 16 on IEEE 118)
  was closed on 2026-10-13 by the spectra-first route without retraining, so
  it is no longer the first RL target; the remaining student regret on
  IEEE 118 is 1.3 calls per root and 20 false commits against the expert.

## 10. Sub-threshold faults: what RL can and cannot do (2026-10-14)

Question (user, 2026-10-14): in the challenging scenarios the WLS decision
threshold is often not reached at all; can RL address those cases?

### 10.1 What the contract does on a quiet WLS today

Three rules make a sub-threshold fault invisible by construction, before any
policy decides anything.

| Rule | Where | Consequence |
| --- | --- | --- |
| Admission requires the operator alarm | `psse_env/wls_admission.py` (`admitted` returns False without `operator_alarm`); `band_admitted` in `research/classifier_triage/ieee118_dataset.py:112`; the detectable-only HIF and unbalance pools | No non-alarming root enters D0, the suites or the classifier datasets, so there are no demonstrations and no labels |
| Every auxiliary stream is refused without a current alarm on the exact state | `psse_env/providers/matpower.py:962` (`diagnostics_require_wls_alarm`), `psse_env/evidence_profile.py:278` | On a quiet state the agent may fetch balanced contexts and finalize; it cannot look at phasors, spectra or branch currents. Healthy roots close in two calls (WLS, finalize) |
| The candidate verifier accepts a correction on alarm-scale progress | the deployment `CandidateQualityOracle`, progress basis `chi_square_excess` or the dof floor | A correction of a sub-threshold fault cannot be verified or committed; the only honest outcomes are a handoff or an acquisition that moves the evidence onto a stream where it is above threshold |

The deployed detector (chi-square at alpha 0.01 or a normalized residual of
4) fires on 2.0% of clean windows (69 of 3,500,
`docs/hypothesis_ranking_step1_20260930.md`) and on 6.5% in the IEEE 57
sweep. Any sub-threshold trigger operates below that line, where healthy
noise is dense, so its cost is acquisitions on healthy roots and must be
reported as such.

### 10.2 The information bound, already measured

No policy, learned or not, can detect from one balanced snapshot what the
snapshot does not contain. The 2026-09-15 structured-screen evaluation on the
IEEE 57 sweep measured the bound directly: the noise-free WLS objective of
the fault, lambda, is the non-centrality any residual-based single-snapshot
test can reach, and the ideal known-direction recall at 1% false triggers is
Phi(sqrt(lambda) - 2.33). The learned-detector control points the same way:
the WLSScreenGNN beat the alarm on unregulated replays (62.7% against 51.9%
phase recall) and not on regulated physics (56.0% against 55.7%, with 3.2%
healthy false triggers), because its edge was a regulation cue, not
sub-threshold information.

| Fault (IEEE 57 sweep, nominal noise) | Deployed alarm | Best single-snapshot statistic | lambda (median) | Snapshots to reach lambda = 10 (about 80% recall at 1%) |
| --- | --- | --- | --- | --- |
| HIF 10 pu | 37% | 72 to 88% | 16.7 | 1 |
| HIF 100 pu | 4% | 4% | 0.17 | about 60 |
| HIF 1000 pu | 4% | 2% | 0.00 | never |
| Unbalance delta 0.05 | 7% | 10% | 0.42 | about 24 |
| Unbalance delta 0.2 | 21% | 42 to 55% | 6.85 | 2 |

The last column assumes a persistent fault, independent noise across
snapshots and a fixed operating point, so that the accumulated statistic's
non-centrality grows as N times lambda. The multi-scan HIF benchmark
(`docs/hif_multiscan_benchmark_20260713.md`) found that more snapshots add
information magnitude rather than new geometry: good for detection, not for
localization.

Where the bound is not binding, the information is on the other streams:
the PMU phasors carry sigma 1e-4 per component against SCADA sigma 0.01 on
P and Q and 0.001 on Vm (`psse_env/fault_profiles.py:37`,
`pipeline.env` `PMU_PHASOR_SIGMA`), and the phase diagnostic localized 224 of
252 HIF roots at 100 pu where the balanced alarm caught 4%. So for weak HIF
and unbalance the detection problem is not "find it in the WLS" but "decide
when to pay for the stream that shows it".

### 10.3 What RL can learn here, and what it cannot

- **Cannot:** raise single-snapshot detectability above Phi(sqrt(lambda) - 2.33).
  Below about one sigma of balanced-channel energy there is no cue, and a
  trigger trained on such roots will fire on healthy noise at whatever rate
  the reward tolerates. These roots are reachable only by a surveillance
  schedule (periodic phasor checks, a budget-allocation problem) or by
  time-series accumulation.
- **Can:** the sequential acquisition policy on a quiet state. Given a
  sub-threshold cue from balanced statistics only (effect ratio between
  about 0.3 and 1.0, the largest normalized residual between 2.5 and 4, the
  structured screen or the triage request score below its threshold,
  multiplier concentration), decide whether to pull another snapshot, which
  stream to acquire, when to stop and declare the window healthy, and
  whether to hand off with a sub-threshold suspicion. This is active sensing
  under a budget: an optimal-stopping problem for one family (SPRT or CUSUM
  over snapshots is optimal there) and a genuine sequential decision problem
  across families and streams, where RL is a legitimate solver. The value is
  in the stop-or-continue and which-stream decisions; a one-dimensional
  trigger threshold is a calibration, not an RL problem.
- **Must be beaten first:** a calibrated sub-threshold trigger at a fixed
  healthy-acquisition rate (an ROC operating point on the GNN or screen
  score), and SPRT over snapshots at the same budget. RL is credited only
  for dominating their recall-against-acquisition curve.

### 10.4 What to build, in order

1. **Sub-threshold draw.** Effect-sampler band (0.2, 1.0) for every family
   (`draw_target` accepts any positive band) with a new admission mode that
   keeps the non-alarming solve and records lambda; healthy windows at
   matched count; a scan window of N balanced snapshots on every root,
   healthy and faulted alike, with independent noise and bounded
   operating-point variation. Phasors, spectra and the scan window on every
   root, so availability never leaks the family (the multi-scan estimator is
   hidden today for exactly that reason, `psse_env/evidence_profile.py:101`).
2. **Detectability table, no learning.** Per root: lambda, the ideal
   single-snapshot recall, the N at which the accumulated statistic reaches
   lambda = 10, whether the NLM and HSE show the fault on the true phasors
   and spectra, and the structured and triage scores against the healthy
   distribution. This is the attainable ceiling per family and effect band,
   the analogue of the identifiability tag of step 1.
3. **Contract.** A `quiet_suspicion` tier that admits one more snapshot
   (`run_wls` on the next scan) and the phasors on a sub-threshold cue; the
   accumulated N-scan statistic as the verification signal for a correction
   on a quiet state; audit bases `subthreshold_fault_confirmed` (the
   acquired evidence names the fault and the correction verifies on the
   stream that shows it) and the healthy declaration after an acquisition,
   which stays a success and carries the acquisition cost.
4. **Baselines.** The calibrated trigger and SPRT over snapshots, each
   reported as recall by family against acquisitions and snapshots per
   healthy root.
5. **RL.** The loop of section 6 on the sub-threshold training draw, with
   healthy roots in every group, reward terms for each extra snapshot and
   each acquisition, and the same graded success. Deployment criterion: the
   RL curve dominates both baselines at the predeclared healthy-acquisition
   budget, and the standard 160 roots do not regress.

What not to do here: no sub-threshold trigger may read anything but balanced
SCADA statistics; no root may carry a scan window or a stream the others do
not; the healthy-acquisition budget is predeclared from step 2, not learned
implicitly through the reward.

## 11. Sources

- Expert end-to-end runs: `output/classifier_triage_20261004/e2e/classifier_baseline_v2`,
  `.../suspicion_ledger_ranked_v2`, `output/classifier_triage_20261013/e2e_ieee14_spectra_first`
  (`episodes.jsonl`: tools, corrections, basis, steps per root).
- Suite strata: `output/classifier_triage_20261004/suites/development.json`
  (`audit.parameter_ranking`); stratum rule in
  `research/hpc/full_pipeline_20260907/build_suite.py:86`.
- Admission filters: `psse_env/examples/generate_round0_aggregate.py`
  (`--admission-mode`), `psse_env/providers/scenario_generator.py:1399` and
  `:2904` (rank allowance), `research/hpc/full_pipeline_20260907/pipeline.env:118-135`.
- Audit bases and score weights: `psse_env/dagger/release_audit.py:2574-2596`,
  `psse_env/dagger/evaluator.py:137-147` and `:5104` (`_tool_regret`).
- Candidate lifecycle: `psse_env/oracle/termination_expert.py:46`
  (`candidate_disposition_actions`: commit on accept, rollback on reject).
- Decoding and training recipe: `psse_env/dagger/research_policy_factory.py:591-595`,
  `psse_env/sft/training.py` (LoRA rank 16 on the attention and MLP
  projections), `pipeline.env` (Gemma 4 12B, BC0 lr 1e-4, rounds 3e-5,
  max length 32,768), `psse_env/requirements-sft-research.txt` (TRL 0.24,
  PEFT 0.19).
- Cell records: `docs/classifier_triage_plan_20261004.md` (sections 10 to
  12), `docs/classifier_triage_ieee118_transfer_20261007.md`,
  `docs/triage_family_scores_20261013.md`, `docs/effect_targeted_magnitude_20261012.md`,
  `docs/hypothesis_ranking_plan_20260930.md` and its step notes.
