# Hypothesis ranking, step 3: the ledger expert and the paired evaluation (2026-10-01)

Step 3 of `docs/hypothesis_ranking_plan_20260930.md`: the rule-based expert
follows the balanced screen's ranked hypotheses, spends a bounded number of
verified candidates, and is compared root by root with the step 2 expert on a
160-root development draw under the research environment's deployment-mode
(observable-only) verifier. Code on `codex/cleanup-20260928`; research only.

## 1. What was built

**The ledger** (`psse_env/oracle/hypothesis_ledger.py`). The balanced screen
already compares the four balanced causes on every alarm, applies the winners
in sequence and ranks the targets inside each class; its compact report rides
on the WLS ledger and now also carries each round's best target and top three
alternatives per class. The ledger reads that report with the state's recovery
records and produces one ranked list of hypotheses (family, target, source,
status: untested, execution_failed, rejected, accepted, budget_exhausted) and
a budget: two verification-rejected candidates per family and four per state.
`ExpertPolicyOracle(hypothesis_ledger=True)` applies it in the combined
proposal stage: the screen's leading family gets a confidence boost of 0.12
(its context fetch and its corrections outrank the other families), the
screen's top-ranked target of that family 0.01 and the second 0.005, a family
or state over budget offers no further correction, and when the leading family
is the meter route the structural-first reordering is not applied (the branch
contexts are not fetched before a meter edit the screen ranked first; the
branch-dominance guard and verification still apply). A failed execution
consumes no budget; a rejection removes its target only; an accepted correction
advances the state and the ledger starts again on the child's own screen.
Everything is read from the policy observation.

**Shared fixes** the first paired run exposed (both arms failed the same four
roots; both arms carry these):

- *A refuted screen explanation opens the phasors.* When every meter or
  branch hypothesis the screen accepted has been tried as a correction on the
  state and rejected by verification, the alarm is unexplained in fact and the
  `phasor` suspicion holds (`screen_explanation_refuted`). The one mixed HIF
  root the screen does not flag (it reads as two bad meters) now reaches the
  phasors after its rejected meter edits, the NLM finds the HIF, and the
  conditioned meter route repairs the meter.
- *The conditioned meter route may retry meters rejected before the HIF was
  found* (`_conditioned_route_seen_signatures`): they were verified against
  the unconditioned model, in which the fault itself holds the residuals up.
- *No re-acquisition after a meter commit.* Phasors or spectra examined on an
  ancestor state in the episode, with no waveform event found, count for the
  descendants (`phasors_examined_in_episode`, `spectra_examined_in_episode`);
  the D3 edit hold reads the same predicate and the harmonic suspicion holds
  on those descendants. The two five-meter roots that spent ten steps per
  meter re-acquiring and ran out of budget now finish in 34 steps.
- *Acquired phasors are examined before anything else*, whatever suspicion
  opened them.

**Tooling**: `research/hypothesis_ranking/expert_e2e.py` (`--expert`,
`--plan`, `--roots-file`), `compare_arms.py` (paired, per family, with the
first-correction hit rate and the differing roots), `debug_root.py` (one root,
full trace with tool errors and the screen's rounds).

## 2. Paired evaluation

160 roots in the development plan of the HPC cell (seed 20261001; generated
once, cached, identical for both arms), 40-step budget, deployment verifier,
suspicion-gated contract. A: the step 2 expert with the shared fixes. B: the
same with the ledger.

| family | n | success A | success B | mean steps A | mean steps B | corrections A/B | rollbacks A/B | phasors | spectra |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_error | 8 | 8 | 8 | 2.0 | 2.0 | 0/0 | 0/0 | 0 | 0 |
| telemetry_no_disturbance | 8 | 8 | 8 | 2.0 | 2.0 | 0/0 | 0/0 | 0 | 0 |
| measurement | 16 | 16 | 16 | 7.5 | 7.4 | 16/16 | 0/0 | 3 | 0 |
| multi_measurement | 12 | 12 | 12 | 23.5 | 19.0 | 45/45 | 0/0 | 3 | 2 |
| parameter | 16 | 15 | 15 | 7.2 | 7.2 | 17/17 | 0/0 | 0 | 0 |
| measurement+parameter | 16 | 16 | 16 | 11.0 | 11.0 | 32/32 | 0/0 | 0 | 0 |
| topology | 16 | 16 | 16 | 8.0 | 7.1 | 16/16 | 0/0 | 0 | 0 |
| measurement+topology | 12 | 12 | 12 | 12.0 | 11.0 | 24/24 | 0/0 | 0 | 0 |
| hif | 16 | 16 | 16 | 6.3 | 6.3 | 0/0 | 0/0 | 16 | 0 |
| measurement+hif | 8 | 8 | 8 | 13.0 | 11.9 | 13/10 | 5/2 | 8 | 0 |
| three_phase_unbalance | 16 | 16 | 16 | 4.0 | 4.0 | 0/0 | 0/0 | 16 | 0 |
| harmonic | 16 | 16 | 16 | 8.4 | 7.6 | 10/6 | 10/6 | 16 | 16 |
| **all** | 160 | 159 | 159 | 8.8 | 8.1 | 173/166 | 15/8 | 62 | 18 |

Acquisitions are identical in both arms (the acquisition rule is shared): 62
phasor requests, 22 of them on roots whose truth needs none (the 3 single-meter
and 3 multi-meter roots with a voltage-channel pick, the 16 measurement+HIF
roots that need them are not counted; the rest are bus-split topology roots
through the unexplained tier), 18 spectra requests, 2 on bus-split roots. No
false commit in either arm. The first correction of a family hits the true
target on 31 of 32 parameter roots and 64 of 64 meter roots in both arms.

What the ledger changes on this draw is cost, not outcome: 0.7 fewer steps per
episode (8.8 to 8.1), 7 fewer corrections and 7 fewer rejected candidates
(15 to 8), concentrated where the screen's sequence disagrees with the static
family order: topology and mixed topology roots skip the parameter context the
screen does not ask for (one step each on 26 roots), harmonic roots try one or
two meter edits instead of two or three before the spectra, mixed HIF roots
try fewer meters before the phasors, and five-meter roots take four steps per
meter instead of six (23.5 to 19.0). On 42 of the 160 roots the two arms
differ in length; on none in success.

The one failure left in both arms is a parameter root whose true line (3-4)
is adjacent to the line both rankings prefer (2-3): the multiplier ranking
calls it dominant and offers one candidate, the screen's refit ranking agrees,
the wrong correction passes verification and is committed. That is the
misranked stratum's identifiability ceiling the plan described, and no
ordering fixes it; a close-alternative comparison would have to be fed a
second candidate the context does not offer today.

Against the very first paired run (before the shared fixes) both arms gained
three roots (156 to 159): the unflagged mixed HIF root and the two five-meter
roots, through the refuted-explanation acquisition and the episode-level
examined rule.

Verification: `psse_env/oracle/test_hypothesis_ledger.py` (5), the step 2
suites, and the broad regression (591 passed, 184 subtests).

## 3. What was deferred, and why

- *Close-alternative comparison before commit* (plan step 3, item 3): the
  verify-both-then-commit protocol costs about six steps per comparison and
  needs the disposition stage to roll back an accepted candidate. On this draw
  the only root it could help is the misranked one, where the second candidate
  is not offered at all; it is left for the ranker step, which will say how
  often the top two hypotheses are close and whether a second candidate is
  worth generating from the screen's ranking.
- *A handoff naming every tested hypothesis* (plan step 3, item 2): the
  existing exhaustion handoff's audit ledger already lists the supported and
  exhausted recovery targets; a dedicated request code was not needed on this
  draw (no episode ended at the budget).
- *Replacing the three special cases* (ambiguous pair, cross-family probe,
  dominance guard) with the ledger: they are safety rules the ledger now
  orders around rather than removes; removing them is a separate ablation.

## 4. What this means for step 4

The screen's ranking is as good as the multiplier ranking on this draw and
cheaper to follow; where both are wrong the alarm is explained by an adjacent
line and verification accepts it. The learned ranker's job is therefore
narrower than the plan assumed: it is not needed to order the families on
IEEE 14, and its two real targets remain the ones step 1 named, the same-sign
flow-meter pair against a true HIF and the voltage-meter error against an
unbalance (both now routed to phasors, so the ranker would save acquisitions,
not outcomes), plus the adjacent-line ambiguity if a second candidate can be
produced for it. The step 1 dataset, re-analyzed with the step 2 screen,
carries the per-class scores, margins and ranked targets for that study.

## 5. Reproduce

```bash
python -m research.hypothesis_ranking.expert_e2e --output-dir output/hypothesis_ranking_20260930/step3_baseline_v2 --expert baseline --seed 20261001 --roots-file output/hypothesis_ranking_20260930/step3_roots.json --plan '{"no_error": 8, "measurement": 16, "multi_measurement": 12, "parameter": 16, "topology": 16, "harmonic": 16, "hif": 16, "measurement+parameter": 16, "measurement+topology": 12, "measurement+hif": 8, "three_phase_unbalance": 16, "telemetry_no_disturbance": 8}'
python -m research.hypothesis_ranking.expert_e2e --output-dir output/hypothesis_ranking_20260930/step3_ledger_v2 --expert ledger --seed 20261001 --roots-file output/hypothesis_ranking_20260930/step3_roots.json --plan '{...same...}'
python -m research.hypothesis_ranking.compare_arms --a output/hypothesis_ranking_20260930/step3_baseline_v2 --b output/hypothesis_ranking_20260930/step3_ledger_v2 --output output/hypothesis_ranking_20260930/step3_compare_v2.md
```

Each arm takes about ten minutes on one core once the roots are cached; the
160-root generation takes about 150 s.
