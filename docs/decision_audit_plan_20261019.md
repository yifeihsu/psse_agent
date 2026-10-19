# Decision-opportunity audit of the hard draw (Gate 2, 2026-10-19)

The RL runs of `docs/rl_experiments_20261014.md` moved the hard-draw score
by at most four roots over its control, all of it on strata whose solution
is a fixed procedure the expert already performs, and never touched the
misranked and mimic roots the draw was built from. Before any further
policy learning, this audit asks the question those runs could not answer:

> At the decision states of the hard roots, does a legal alternative exist
> that leads to a better outcome, is it predictable from what the policy
> observes, and when none exists is it the action interface or the evidence
> that is missing?

It is the review's Gate 2. It runs on the CPU, with the environment, the
observable expert and the evaluator the runs already use, and it produces a
four-way classification of the decision states plus the data for a
predictability test. It does not train anything.

## 1. The objects

**Reference trajectory.** The observable expert (`ObservableExpertPolicy`
over `ExpertPolicyOracle`, variant `ledger_ranked` under the
`suspicion_gated_diagnostics` profile, the screen cell's teacher) rolled
out on the root through `evaluate_rollout_suites`, which records the policy
observation, the action and the outcome of every step and audits the
terminal state against the hidden truth. A second source, the student's
own recorded trajectories from the stored RL rollouts, replays the same
way (phase 2).

**Decision state.** A policy step of the reference trajectory at which the
observable menu holds at least two distinct admissible actions and the
expert's choice is a rule-expert choice rather than a candidate
disposition it is obliged to take (commit or rollback after a verdict, the
verification of an open candidate, the opening solve). The menu is the
expert's own ranked proposal list recomputed from the stored policy
observation (`select_observable_expert_actions`), so every alternative is
one the deployed policy could have produced from the same evidence; hidden
truth never enters the menu. At most the first K = 4 decision states of a
root are audited, and at most the top three alternatives at each.

**Branch.** A fresh evaluator episode on the same root whose policy replays
the reference actions up to the decision state, takes the alternative, and
then follows the observable expert to the end. The branch is checked
against the reference at the decision state (same observation hash) and
scored by the evaluator exactly as the runs are scored: truth-audited
success and its basis, false commits, invalid actions, loops, steps, and
the graded episode reward of the RL contract.

**Truth branch.** At the first decision state whose menu offers a
correction, one more branch takes the correction built from the root's
hidden truth (all true targets at once: the true line, or the true meter
group as one `suspect_group`), then follows the expert. It is not a legal
action of the deployed policy; it is the audit's probe of the interface and
the evidence, the identifiability tagger's test taken at the policy's own
state instead of the fresh one.

## 2. The classification

Per decision state, with R the reference outcome, A* the best legal
alternative and T the truth branch:

| Bin | Condition | Reading | Next action |
| --- | --- | --- | --- |
| `rule_best` | R succeeds and no legal alternative succeeds with fewer charged steps | the fixed rule already picks the best alternative | nothing to learn here |
| `legal_alternative` | R fails and some legal alternative succeeds, or R succeeds and an alternative succeeds cheaper | a genuine ordering or acquisition decision | the predictability test; then train it |
| `interface_limit` | R and every legal alternative fail, T succeeds | the right correction is not in the menu (joint pair, absent candidate) | repair the interface and give it to every baseline |
| `evidence_limit` | R, every legal alternative and T fail | the verifier cannot separate the hypotheses at this state | add a justified observation or score honest abstention |

A root's bin is the best bin of its audited states; a root whose reference
succeeds and shows no cheaper alternative is `rule_best` outright.

**Predictability** (phase 3). For every audited decision state the audit
stores the features of each alternative as the policy sees them: rank in
the menu, tool, target, the fresh context evidence (parameter-ranking
multipliers and dominance ratio, measurement findings), the remaining
budget, the counts of accepted corrections and rejected hypotheses, the
step index. A gradient-boosted ranker trained on other physical parents
(parent-aware cross-validation, never on development roots) predicts which
alternative wins; its accuracy against the fixed rule ("take the first")
is the learnability of the `legal_alternative` states. An oracle that
chooses after seeing the branches is retrospective selection and is not
the quantity reported.

## 3. Sample size

The training draw has 256 roots; the expert solves 242. Its hard strata
are small: 8 misranked parameter roots (expert 2 of 8), 48 flow-meter
mimics (expert 43), 72 ambiguous parameter roots (expert 70), 32
multi-meter roots (expert 31). The development draw, which stays held out,
has 17 misranked roots.

Thirty-two roots are enough for the first purpose of the audit, which is
to see the bins fill and to check the machinery: all 8 misranked roots, 12
mimics (6 of each sign), 8 ambiguous parameter roots preferring the ones
the identifiability tagger marked `not_separable`, and 4 controls (2
harmonic, 2 multi-meter). At four states and three alternatives each that
is at most 32 × 13 episodes of about 4 s, under half an hour on one CPU.

They are not enough for the second purpose. The predictability test needs
decision states in the hundreds with parents held out, so after the pilot
the audit runs on every training root of the failure-prone strata (the 8
misranked, 48 mimics, 72 ambiguous, 32 multi-meter and 8 dominant: 168
roots, about three hours on one CPU, or minutes across cluster cores),
and the ordering decisions of all 96 parameter-bearing roots feed the
ranker whether or not the expert fails on them, because the label (which
candidate is true) is known for each. The misranked stratum itself has 8
training parents, so any claim about it rests on the audit's bins and on
the development set's 17 roots scored once, at the end, by a ranker that
never saw them.

## 4. Deliverables

- `research/rl/decision_audit.py`: the driver; `states.jsonl` (one row per
  audited decision state with its branches and features), `roots.jsonl`,
  `summary.json` and `summary.md` (bins by stratum, costs, the truth-branch
  outcomes).
- Phase 2: the same driver with `--prefix-source rollouts`, replaying the
  student's recorded action sequences from the RL trajectory store, so the
  audit covers the states the deployed policy visits, not only the expert's.
- Phase 3: `research/rl/decision_ranker.py`, the parent-aware predictability
  test over `states.jsonl`.
- A section in the run note with the bin table and the decision it implies
  (Gate 3 if `legal_alternative` is populated and predictable; interface or
  evidence work first otherwise).
