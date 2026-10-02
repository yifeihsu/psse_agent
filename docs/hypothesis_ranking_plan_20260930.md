# Ranked hypothesis testing on balanced evidence: decision and experiment plan (2026-09-30)

Answer to the two questions of 2026-09-30 (rank error hypotheses and test them in
order; learn the family ranking), read against the reviewed external answer and
against the code on `codex/cleanup-20260928` (from `codex/hif-gate-20260927`,
suspicion-gated contract of 2026-09-27). Research plan only; nothing here has
been run.

## 1. Decision in one paragraph

Yes to both, in this order: (1) make the balanced physics screen the hypothesis
ranker it already almost is, add an explicit "no single balanced cause explains
the alarm" outcome, and measure offline what balanced evidence can and cannot
separate; (2) generalize the expert's existing outcome-dependent switching
(ambiguous-branch testing, cross-family diversification, branch-dominance guard)
into one bounded sequential controller over that ranking, and evaluate it with
the rule-based expert on CPU; (3) train a learned ranker on the same offline
dataset and deploy it only if it beats the physics screen on held-out parents;
(4) only then rebuild the teacher, re-collect D0, and run the DAgger cell. The
external answer's architecture (rank, investigate, test a candidate, interpret,
rerank; never treat a failed commit as a negative family test) is right and is
adopted. Its Stage A is half done already, its Stage B is the cell we have, and
two of its three repository claims are stale (section 7).

## 2. What the current pipeline actually does when evidence is insufficient

Read from the code, not from memory of earlier contracts.

| Situation after a WLS alarm | What happens now | Where |
| --- | --- | --- |
| Balanced screen: split-line shunt wins the penalized chi-square comparison | `wls_hif_suspected line=L` is minted; phasors (present on every root at 1e-4) are admitted; NLM confirms or refutes on the zero-sequence differential; a refutation refreshes WLS and opens the balanced routes | `psse_env/providers/hif_screen.py`, `matpower.py:1110-1160`, `matpower.py:1941-1949`, `diagnostics_expert.py:suspicion_screening_proposals` |
| Balanced screen: meter, parameter or topology wins | Only the `suspected` flag is used. The per-class scores ride on the WLS ledger (`hif_screen.rounds[].scores`) and are policy-visible, but no expert reads them; routing falls back to the residual/multiplier dominance tags (ratio 1.2) | `matpower.py:1846-1875`, `expert_policy.py:350-420` |
| Top parameter line does not dominate its runner-up (ratio < 1.2) | Top-2 lines tested under verification in rank order, then a handoff bounded to those two lines (`operator_escalation:ambiguous_branch_candidates`), scored as success when the true line is among them | `expert_policy.py:1053-1140`, `release_audit.py:2509-2515` |
| Branch fix clears its own multiplier but global progress < 30% | Meter route probed instead of the next-ranked line | `expert_policy.py:_cross_family_diversification_proposals` |
| Branch-dominant solve | Meter corrections refused until both branch families have a rejected hypothesis on this state | `matpower.py:2270-2285` |
| Every proposal filtered out | `operator_escalation:recovery_options_exhausted` | `expert_policy.py:_recovery_exhaustion_proposals` |
| Unbalance or harmonic root | No screen produces their suspicion, so phasors and spectra are refused; the balanced ladder runs, corrections are rejected, the episode ends in the generic handoff, and the audit scores it as a failure (the only handoff credited is the bounded branch one) | `evidence_profile.py:DIAGNOSTIC_SUSPICION_REQUIREMENTS`, commit 24ab480 message, `release_audit.py:audit_truth_audited_task_success` |

Two facts shape everything below.

- **The candidate verifier the expert sees is observable-only.** The
  transactional environment passes `hidden_truth` into
  `CandidateQualityOracle.label_candidate` (`transactional_env.py:3738-3743`),
  but the research and release environments build that oracle in deployment
  mode (`MatpowerDeploymentProviders._deployment_candidate_quality_oracle`,
  asserted by `resolve_environment_factory`), which ignores the truth. Only an
  environment built with a default `CandidateQualityOracle()` (mode `auto`, the
  unit fixtures) reads it. So when the expert "tests the second-ranked error
  after the first is rejected", the rejection is a verdict the student would
  also see, and the sequential experiments below are measured under the
  deployment verifier by construction. (Corrected 2026-10-01; the first
  version of this document had it the other way round.)
- **The 2026-09-26 feasibility study already answered the HIF half of Stage A.**
  On 1,324 alarmed IEEE-14 roots the balanced screen flags 130/131 HIF and
  130/131 measurement+HIF with the faulted line right in every flag and no
  non-HIF root flagged (commit 24ab480; held-out half: 64/65 and 2/1,062). The
  same study found that unbalance and harmonic roots are separable from meter
  and parameter errors on balanced SCADA only through simulator artifacts
  (OpenDSS negative-sequence background, the bus-9 resonance). That is an
  identifiability limit, not a weak rule set. No ranker trained on balanced
  evidence will recover those two families honestly; what a ranker can add is
  the *unexplained* class and better target-level ranking on mixed roots.

The current development ceiling follows directly: 16 unbalance and 16 harmonic
development roots out of 160 cannot succeed under the contract as it stands,
so the expert's ceiling on the 2026-09-27 contract is 128/160 unless the
scoring or the acquisition rule changes. That is the first thing to fix.

## 3. Design conflicts to decide before building

Stated plainly, as required, because each one changes what gets built.

**C1. What opens phasors on a root the HIF screen does not flag.** The standing
rule is "three-phase data only on a WLS/NLM HIF suspicion". Unbalance roots
have no HIF suspicion and no other balanced signature. Three options:

- (A) *Recommended.* Extend "suspicion" to `hif OR unexplained`, where
  `unexplained` means the best single balanced cause, after the two-round
  meter set-aside, still leaves the operator vector alarmed (J above its
  chi-square threshold or max normalized residual at or above 4). This is
  evidence that the balanced hypotheses were tested by refit and none fits. It
  is not "corrections failed, so it must be HIF": no commit is involved, the
  refits are cheap, and the phasors then decide (zero-sequence differential
  for HIF, shunt-power spread for unbalance, balanced three-phase for neither).
  Phasors exist on every root, so availability cannot leak the family. The
  cost is extra acquisitions on multi-meter and mixed roots and on false
  alarms of healthy roots; Step 1 measures that rate before anything is wired.
- (B) Keep the rule strict and add an audit rule that credits a
  correction-free handoff on a root whose family has no admitted route (no
  healthy component modified, no false partial claim). Honest, but it turns
  32 of 160 development roots into "correct handoffs" by definition and the
  agent learns nothing about them.
- (C) Build a balanced unbalance screen. The feasibility result says it would
  rest on simulator artifacts; not recommended.

**C2. Spectra.** The rule is "spectra only on a harmonic suspicion", and no
balanced screen provides one. Under (A), harmonic roots will reach
`unexplained`, acquire phasors, and the phasors will come back balanced (the
harmonic corpus is a fundamental-frequency synthesis). The choice is whether
"unexplained by balanced causes and by phase-resolved phasors" may open spectra
as a second-tier acquisition. Recommendation: yes, but only as the last rung and
only measured; if the user prefers to keep spectra closed, harmonic roots take
the (B) handoff rule and are reported as such.

**C3. Verifier mode for the sequential experiments.** Resolved by the
correction above: the research environment already verifies in deployment
mode, so Stage B is measured under observable verification. The truth enters
only the offline audit.

**C4. What the ranker is allowed to consume.** Only what the policy sees:
the WLS ledger (chi-square, normalized residuals with sign and channel type,
normalized multipliers per branch, dominance tags), the screen report
(per-class scores, margins, J after each best refit, set-aside channels),
model connectivity, and the same-state action history. Never the fault
label, `op_point`, corpus identity or any auxiliary stream before it is
admitted. This is the existing strict-boundary denylist and needs no new rule.

## 4. Experiments

### Step 1 (CPU, about two days): offline hypothesis dataset and identifiability report

Purpose: answer "can the balanced evidence distinguish the faults, and where
does it stop" with numbers, on the current physics, before touching the
expert.

1. Add to `screen_hif` (no behaviour change to the policy path yet):
   `J_after` and the alarm test for every class's best refit, the score margin
   between the top two classes, and an `unexplained` outcome when no class's
   best refit clears the alarm after the two rounds. Keep the compact
   policy-visible report unchanged for now.
2. Build the dataset with `Round0ScenarioGenerator.build` over all 12
   families under `suspicion_gated_diagnostics` with the 20260923opf corpora,
   about 250 roots per family (about 3,000 roots; the screen costs about
   0.3 s per round on one core). For each root store the WLS ledger features,
   the full screen report, and the truth (family set, meter indices, branch
   rows, HIF line and alpha). Split by physical parent (corpus row or OPF
   operating point), never by noise replica; hold out a calibration and a
   test partition of parents.
3. Also generate trajectory states: for measurement+parameter,
   measurement+topology and multi_measurement roots, apply the true first
   correction and rerun WLS and screen on the child state, labelled with what
   *remains*. This is the "what should be fixed next" label the external
   answer asks for, and it is cheap because the truth correction is known
   offline.
4. Report, per family and per stratum:
   - confusion of the screen's final winner (meter, parameter, topology, hif,
     unexplained) against the truth family set;
   - the rate at which `unexplained` captures unbalance, harmonic and
     multi_measurement roots, and its false rate on single-cause roots and on
     alarmed no_error roots (this is the acquisition cost of option A);
   - top-k target recall inside the winning class (meter index, branch row),
     against the existing residual and multiplier rankings;
   - mixed-root coverage: whether the two-round outcome names both components;
   - the known mimic: two bad flow meters at both ends of one line versus a
     real HIF, on purpose-built roots.
5. Repeat the screen-only part on the IEEE-57 balanced families plus its HIF
   and unbalance corpora (`output/ieee57_hif_unbalance_20260911_verified`,
   `output/ieee57_physical_hif_20260919`) so the transfer question is asked
   early. Expect the R at or above 100 pu HIFs to be non-alarms rather than
   confusions (the balanced-screen bound of 2026-09-10).

Decision rule at the end of Step 1: if `unexplained` captures at least 80% of
unbalance and harmonic roots at under 5% false rate on single-cause roots,
option A is viable and Step 2 wires it; otherwise take option B and reconsider
the contract.

### Step 2 (one to two days): acquisition rule v2 and the handoff scoring rule

1. Mint a second policy-visible signature, `wls_unexplained_balanced_discrepancy`,
   when the screen ends `unexplained`; let `required_suspicion` accept it for
   phasors (and, under C2, for spectra once the phasor NLM has reported
   `balanced_three_phase` on the same state). The environment already
   classifies acquired phasors as HIF, unbalance or balanced
   (`matpower.py:4860-4905`), so the unbalance ladder works unchanged once the
   stream is open.
2. Add the audit basis `unexplained_discrepancy_handoff` in
   `release_audit.audit_truth_audited_task_success`: a correction-free handoff
   after an `unexplained` screen counts as success only when the truth family
   is one the contract could not diagnose and no healthy component was
   modified. Report it separately from `counterfactual_resolution` and
   `bounded_localization_handoff`, never merged into one number.
3. Rerun the local end-to-end expert check of commit 24ab480 with unbalance
   and harmonic roots included; record the acquisition counts per family.

### Step 3 (CPU, three to four days): bounded sequential controller in the expert

Replace the three special cases (ambiguous branch pair, cross-family
diversification, branch-dominance guard) with one hypothesis ledger per active
state, without changing what the candidate verifier does.

1. Ledger entries: family, target (meter index or branch row), score from the
   screen or ranker, status (untested, execution_failed, inconclusive,
   rejected_locally_fixed, rejected, accepted_partial), and the verification
   metrics. The outcome typing follows the external answer's table: an
   execution failure keeps the score, an inconclusive test records the
   limitation, a rejection removes only that target, a locally fixed but
   globally insufficient candidate promotes the competing family.
2. Budget: at most two verified candidates per family and five per episode
   before a bounded handoff that names every tested hypothesis
   (`operator_escalation:ranked_candidates_exhausted`, replacing the generic
   exhaustion request when a ledger exists). Predeclare these numbers; do not
   tune them on the development roots.
3. Close alternatives before committing: when the top two hypotheses are
   within the screen's discrete penalty of each other (0.25 ln candidates)
   and both are admissible, verify both on the same parent state and commit
   the one with the larger untargeted consistency gain, not the first one that
   passes. This is the safeguard against accepting the wrong explanation; it
   costs one extra verification only on close calls.
4. Reranking after a partial commit re-runs the screen on the child state
   (the ledger is per state, so this is automatic).
5. Evaluate on the 160 development roots with the rule-based expert under
   the deployment verifier (C3), two arms: the current expert and the ledger
   expert.
   Primary outcomes: truth-audited success by basis, false commits, healthy
   components modified, unnecessary acquisitions (phasors or spectra on a root
   whose truth did not need them), steps to terminal. Secondary: the
   ambiguous and misranked parameter strata, where the ordering matters most.

### Step 4 (GPU hours, in parallel with Step 3): learned ranker on the Step 1 dataset

1. Baselines: the physics screen's own class scores (no learning); a
   gradient-boosted model on screen plus WLS features; the existing
   `WLSScreenGNN` retrained on the same manifest through
   `research/gnn_screen/dagger_corpus.py` (its family heads and calibration
   code already exist; it is blocked at runtime under strict profiles and
   stays blocked until this comparison is done).
2. Targets: family-presence marginals (measurement, parameter, topology, hif,
   unexplained) that need not sum to one, and per-family target ranks. Train on
   the train parents, calibrate on the calibration parents (temperature
   scaling), report reliability diagrams and parent-bootstrap intervals on the
   test parents.
3. Deployment criterion: the ranker replaces the screen scores in the ledger
   only if it improves mixed-root coverage or the HIF-versus-flow-meter-pair
   confusion by a margin whose parent-bootstrap interval excludes zero, at
   the same false-acquisition rate. If it does not, the physics screen stays
   as the ranker and the learned model is reported as a negative result. The
   2026-09-22 GNN result (56.0% versus 55.7% phase recall on regulated physics)
   is the prior here: expect the gain, if any, to be on ranking rather than
   detection.
4. Report calibrated probabilities as simulation-distribution probabilities
   with the declared class mixture, not as operating probabilities.

### Step 5 (HPC cell, after Steps 2 to 4 have numbers): closed-loop comparison with the LLM

The external answer's four arms, on the existing cell
(`research/hpc/full_pipeline_20260907`, suspicion-gated default):

| Arm | Teacher or policy | What it isolates |
| --- | --- | --- |
| 1 | Current expert (2026-09-27 contract plus Step 2 scoring) | Baseline ceiling with honest unbalance and harmonic scoring |
| 2 | Ledger expert with screen scores (Step 3) | Value of bounded sequential testing without learning |
| 3 | Ledger expert with the learned ranker (Step 4), if it passed | Value of data-driven discrimination |
| 4 | Gemma student trained by DAgger on arm 2 or 3 demonstrations | Whether the LLM keeps the teacher's decisions and recovers from its own errors |

Arms 1 to 3 need only CPU and can be paired on the same 160 roots before the
cell is submitted. Arm 4 is the usual chain (D0 548 roots, BC0, two rounds of
122 roots, paired evaluation) on the big-GPU pool with preemption opt-in;
D0 must be regenerated because the teacher changes. Report full-pipeline
recall separately from success conditional on a WLS alarm, so the detector's
misses stay visible, and report success by basis
(`counterfactual_resolution`, `bounded_localization_handoff`,
`unexplained_discrepancy_handoff`).

## 5. What not to do

- Do not feed the episode's true family, or anything derived from it, to the
  teacher at execution time; supervise the ranker offline and let the teacher
  consume only the ranker's outputs (the observable-teacher protocol stands).
- Do not let HIF, or `unexplained`, become the default explanation after
  commits fail. Only refit evidence and acquired measurements move a family
  up; execution failures and budget exhaustion never do.
- Do not multiply likelihoods across repeated deterministic solves on the same
  measurements; one WLS ledger per state is one observation.
- Do not tune the ledger budget or the penalties on the development roots.
  The screen penalties were tuned on the design half of the 2026-09-26 set;
  the Step 1 test partition is new parents.
- Do not report a union of screens using one screen's false-trigger rate.

## 6. Order and effort

| Step | Depends on | Effort | Output |
| --- | --- | --- | --- |
| 1 Offline dataset and identifiability report | nothing | 2 days CPU | `output/hypothesis_ranking_20261001/` with the report and the parent-split manifest |
| 2 Acquisition rule v2, handoff scoring | decision on C1 and C2, Step 1 numbers | 1 to 2 days | code on a new branch from `codex/cleanup-20260928`; local e2e on all 12 families |
| 3 Ledger expert, CPU arms 1 and 2 | Step 2 | 3 to 4 days | paired 160-root report under the deployment verifier |
| 4 Learned ranker | Step 1 | 1 day plus GPU hours | calibrated ranker report, deploy or negative result |
| 5 HPC cell, arms 3 and 4 | Steps 2 to 4 | one cell (about a week wall clock) | DAgger rounds and paired evaluation |

## 7. Corrections to the external answer's repository claims

- `RecoveryExpert` is the process-failure repair expert (invalid arguments,
  stale state references); the verify, commit and rollback lifecycle lives in
  `TerminationExpert.candidate_disposition_actions` and
  `CandidateQualityOracle`. The conclusion is unaffected.
- `psse_env/dagger/aggrevate.py` was removed on 2026-09-28 (commit cf0c5b3)
  with the DAgger-1 study machinery. The counterfactual generator
  (`psse_env/dagger/counterfactual_generator.py`, three isolated branches per
  root) is the surviving scaffold for comparative outcomes.
- The 724b0c3 snapshot predates the suspicion-gated contract; the balanced
  physics screen the answer proposes as a future ranker already exists and is
  verified, so Stage A starts from its class scores rather than from scratch.

## 8. Step 1 outcome (2026-09-30)

Decisions taken by the user on 2026-09-30: C1 = option A, C2 = yes. Step 1 ran
the same day; results in `docs/hypothesis_ranking_step1_20260930.md`, data in
`output/hypothesis_ranking_20260930/`.

- The shipped HIF suspicion reproduces on a fresh draw: 130 of 131 HIF roots
  flagged on the true line, none of 1,474 alarmed non-HIF roots flagged.
- `unexplained` as a first-alarm trigger captures 80.6% of unbalance plus
  harmonic roots but fires on 18.6% of single-cause roots, so option A as
  written fails the decision rule. The false rate is three screen defects
  (single-parameter refit on roots with both R and X wrong; rounds that stop
  after a physical win; no bus-split class), each measured and each fixable;
  with them fixed the false rate is about 1% and the capture about 70%.
- The unbalance roots that remain explained are all explained as a bad
  phase-A voltage meter (48 of 48). A leading voltage-meter hypothesis as an
  additional phasor trigger would capture 161 of 162 unbalance and 249 of 250
  harmonic roots at the cost of an acquisition on about 13% of single-meter
  roots. This is decision D3, pending.
- On the IEEE 57 study roots the HIF class rarely wins against the
  voltage-meter explanation (16 of 112 alarmed HIF roots), and case57 gives
  the screen no candidate lines by default.

Step 2 therefore starts with the three screen changes and implements option A
as the ledger's terminal condition rather than a first-alarm rule; D3 waits
for the user.

## 9. Step 2 outcome (2026-09-30)

Decision D3 approved; step 2 implemented the same day
(`docs/hypothesis_ranking_step2_20260930.md`). The screen gained the joint
R/X variant and rounds that continue after a parameter or topology win (HIF
flags unchanged at 130 of 131 with no false flag; parameter misses 47 to 0;
mixed roots naming both components 92 to 243 and 22 to 248 of 250). Phasors
now open on an HIF win, a voltage-meter pick or an unexplained alarm; spectra
once phasors come back balanced; a voltage-meter edit waits for the phasors;
every root carries phasors and noise-only spectra; the ladder's honest end is
`operator_escalation:unexplained_balanced_discrepancy`, credited by the audit
basis `unexplained_discrepancy_handoff`. The expert resolves all 48 of 48
fresh roots across the twelve families (unbalance in 4 steps, harmonic in 6),
where before the step the 32 unbalance and harmonic development roots could
not be credited. Step 3 (the ledger expert and the observable-verifier
evaluation) starts from here.

## 10. Step 3 outcome (2026-10-01)

The ledger expert (`docs/hypothesis_ranking_step3_20261001.md`) follows the
screen's accepted sequence and ranked targets and caps verified candidates at
two per family and four per state. On a 160-root development draw under the
deployment verifier it matches the step 2 expert on outcome (159 of 160 each;
the one failure is the adjacent-line parameter root both rankings misrank) and
costs less: 8.1 against 8.8 steps per episode, 8 against 15 rejected
candidates, no false commit in either arm. Two shared fixes found by the
paired run (a verification-refuted screen explanation opens the phasors; no
re-acquisition after a meter commit) took both arms from 156 to 159. The
close-alternative comparison and a dedicated ranked-candidates handoff were
deferred; the learned ranker (step 4) has a narrower job than planned.

## 11. Step 4 outcome (2026-10-01)

The learned ranker (`docs/hypothesis_ranking_step4_20261001.md`) trained on
the step 1 states with parent splits reaches AUC 98 to 100% on every family,
removes the same-sign flow-meter mimic entirely at the screen's HIF recall
(-37.5% [-61.1, -15.8] mimics flagged, one HIF root in fifty lost), separates
an unbalance from a voltage-meter error better than expected (AUC 92.7%
[83.6, 99.1] on 38 roots), gains nothing on the adjacent-line ambiguity, and
trades recall for false acquisitions rather than dominating the shipped gate
(1.2% against 12.0% unnecessary acquisitions at 96.9% against 98.5% recall).
Its IEEE 57 operating points do not transfer. Decision: the physics rule stays
the admission gate; the learned scores become an ordering signal for the
ledger expert (try the leading balanced hypothesis before an acquisition the
model deems unlikely to be needed), to be measured with the step 3 paired
harness. Step 5 (the DAgger cell) can start from the ledger teacher.

## 12. Step 5 outcome (2026-10-01)

The learned ordering is built (`docs/hypothesis_ranking_step5_20261001.md`):
`psse_env/oracle/learned_ranker.py` computes 73 policy-visible features
(parity with the offline study on 13 fresh roots, every feature equal), the
step 4 gradient-boosted models are exported as JSON trees
(`psse_env/oracle/models/learned_ranker_ieee14_20261001.json`, checked
against sklearn to 1e-16) and `ExpertPolicyOracle(hypothesis_ledger=True,
learned_ranker=...)` defers an admitted phasor acquisition behind the
ledger's leading balanced hypothesis, once per state, when the model's
`needs_aux` probability is below the operating point that keeps the rule's
recall. Building it surfaced a contract conflict: the screen's HIF suspicion
carried the HIF marker, so the process gate, the provider contexts and the
expert's combined stage all closed the balanced routes while it stood, and
the deferral was inadmissible. Decision C5: an untested screen suspicion (no
phasors requested on the state) is not a waveform signature; once phasors are
requested it blocks as before; sensor-reported and phasor-confirmed
signatures always block. The step 3 arms reproduce root for root under C5.

Paired on the 160 development roots plus 48 flow-meter pair mimics: arm 3
(ledger + ranker) equals arm 2 in success, corrections and rollbacks and
spends 63 instead of 72 acquisitions, the nine saved being same-sign mimics
with an HIF-won screen; no auxiliary root is deferred. Arm 2 and arm 3 lose
four opposite-sign mimics to arm 1 (17 against 21 of 24): the screen's
accepted sequence puts a branch hypothesis ahead of the meters there, an
open item for the screen. Arm 4 is configured
(`research/hpc/full_pipeline_20260907/overrides/hypothesis_ranking_20261001.env`,
teacher `ledger_ranked`, everything regenerated; local stage 0 passed) and
was submitted on 2026-10-01 (cell research_full_pipeline_20261001_ranked,
jobs 18967774 to 18967781, commit 7f8fb6a). Stage 0 failed on a voltage-meter
root the screen explained by a branch: the D3 hold had no admissible
acquisition and the ledger budget handed off with supported targets
outstanding. Fixed (D3 holds only under a current phasor suspicion; the
budget ranks a family last instead of dropping it, with an open acquisition
tier taken first) and resubmitted from dbc2f88 as jobs 18987065 to 18987076.
That stage 0 completed the aggregate (514 roots) and died in the suite
builder's module loader (fixed in 2a60b68); the suites were built by job
18996296, bc0 then tripped the suite receipt's missing variant declaration
(fixed in 48c5407), BC0 trained (838 steps, eval loss 0.0019), the BC0
receipt tripped the same check (fixed in 46210dc), and the chain resumed at
r1c as jobs 19022934 to 19022939. See the step 5 note.

Round 1 under the honest contract: expert 160, BC0 152, R1 158 of the 160
development roots. Round-2 collection then stopped on a pure HIF root whose
residual outlived the accepted estimate: the expert's unexplained-discrepancy
handoff ignored that the phasors had named the event (fixed: the request
mirrors the audit). An off-path probe found refused and off-target diagnostic
calls hiding the expert's own rung (fixed: a refusal tests nothing, the
estimator rung is judged on the localized line, an acquisition on the active
state counts through its ledger). Expert labels on every stage-0, round-1,
round-2 and development path are unchanged. Fixed in a76f704 and resumed at r2c as jobs 19052385 to 19052388.
Round 2 completed on 2026-10-02: R2 156 of 160 against R1 158. The three new
misses all follow balanced phasors on a voltage-meter suspicion, where the
teacher edits the meter on some roots and asks for spectra on others; the
next teacher asks for spectra first.
