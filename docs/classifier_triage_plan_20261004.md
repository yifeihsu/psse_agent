# Classifier triage: replacing the exhaustive screen (plan and stage 0, 2026-10-04)

## 1. Why

The suspicion-gated contract (2026-09-27 to 2026-10-02) made detection honest:
every root carries phase-resolved phasors and spectra, and a request needs a
balanced reason. The reason came from the balanced hypothesis screen
(`psse_env/providers/hif_screen.py`), which refits the operator model under
every single cause: 229 constrained WLS refits per round on IEEE 14, a median
of 1.1 s per alarm against 0.05 s for the WLS itself, and a refit count that
grows with the branch count. The agent then read the screen's verdict in its
prompt, so the LLM never classified anything from the WLS result.

Direction (user, 2026-10-04): replace the exhaustive screen by an auxiliary
classifier that reads the balanced WLS result only, and compare two forms of
it: a graph neural network meant to move across network sizes without
retraining, and the LLM itself deciding from the WLS result.

## 2. Decisions (user, 2026-10-04)

| | Decision |
| --- | --- |
| G1 | The classifier's thresholded score gates a request for phase-resolved measurements. |
| T1 | Triage labels are truth-derived: needs phasors when the truth holds an HIF, an unbalance or a harmonic source; the balanced families present; on a mixed root the family whose removal lowers the WLS objective most comes first. |
| S1 | Start with the IEEE 14 offline benchmark; roots for IEEE 57 and 118 come later. |

How G1 reads in each arm. GNN arm: the GNN's request score at or above a
threshold set on the calibration split admits a phasor request. LLM arm: the
LLM is the classifier, so its own decision is the gate, and the policy
wrapper can threshold the probability of the request action when a tunable
operating point is needed; offline it is scored both ways (section 6).

Two rules of the current contract do not depend on the screen and stay as
fallbacks in both arms, because a thresholded classifier has no recovery of
its own after a miss: phasors open when every balanced correction offered on
the state was rejected by verification or none is offered while the alarm
stands, and spectra follow phasors that came back balanced.

## 3. What a triage classifier decides

Two outputs are decisions: request phase-resolved measurements or not, and
which balanced family to investigate first. A six-way fault type is reported
as a diagnostic only. Unbalance, harmonics and a bad voltage meter are not
separable on balanced data except through properties of the simulator
(`docs/hypothesis_ranking_plan_20260930.md`, contract review), and the HIF
line does not have to come from the
classifier: the zero-sequence line differential on the acquired phasors
localizes it.

## 4. Stages

| Stage | What runs | Cost | Status |
| --- | --- | --- | --- |
| 0 | Offline benchmark on the IEEE 14 study rows: screen rule, gradient-boosted models, GNN, LLM | CPU and one local GPU; the LLM leg needs a cluster fine-tune | done (section 6): one training pass per LLM variant, then three passes |
| 1 | New evidence profile without the screen; rule expert driven by each triage source on the 160 development roots | CPU | done 2026-10-06 (section 10): classifier profile 159/160 in 4.1 min, screen profile 159/160 (baseline) and 160/160 (ledger teacher) in about 10 min |
| 2 | One DAgger cell per arm | about a day each | GNN arm launched 2026-10-07 02:02 UTC (section 11); no cell for the LLM as the classifier (section 7, item 8) |
| later | IEEE 57 and 118 roots of every family; leave-one-network-out | generation plus training | not started |

## 5. Stage 0 design

Code: `research/classifier_triage/` (`data`, `features`, `model`,
`train_gnn`, `benchmark`, `location_study`, `transfer_look`, and for the LLM
arm `llm_dataset`, `llm_score`, `prompt_control`), tests in
`research/classifier_triage/tests/`, cluster jobs in
`research/hpc/classifier_triage_20261004/`.

**Rows.** The step-4 study rows
(`output/hypothesis_ranking_20260930/ieee14_v3`): alarmed states with a valid
screen, split by physical parent into train 2,277 (with truth-corrected
children), calibration 458 and test 463 (130 need phasors, 333 need none,
including alarmed healthy windows and flow-meter-pair mimics).

**Evidence.** The operator's balanced WLS solve of the state: signed
normalized residuals of every channel and normalized Lagrange multipliers of
every branch's R and X.

**GNN.** One node per bus, two directed edges per branch, three
message-passing blocks with mean and max aggregation and pooling (the blocks
of `research/gnn_screen`), 706,506 parameters, nothing tied to the network
size. Heads: request, six families, first balanced family. Five seeds,
scores averaged. The default view is residual-only: residuals, multipliers,
the configured network. Observed and fitted values, the estimated state and
the sigmas are not read (the `values` view adds them as an ablation).

**Tabular references** (the step-4 gradient-boosted model on different
inputs): the screen's outputs; 45 WLS-only study features; what the agent's
prompt shows of a WLS solve today (the five largest residual magnitudes and
five largest multipliers); richer prompts (ten or twenty residuals with
signs); every residual and multiplier as one flat vector.

**Operating point.** Every learned classifier is thresholded on the
calibration split at the screen rule's calibration recall (99.3%), so the
table compares unneeded requests at equal recall.

**Cues neutralized or measured.**

- *Corpus format.* The topology roots treat the zero-injection bus as exact
  rows and carry a different sigma vector; no other root does. The features
  never read sigmas and zero the injection residuals of zero-injection buses
  on every row.
- *Simulator background.* Every waveform root comes from the OpenDSS corpora
  and every balanced root from the OPF corpus. The probe measures whether a
  classifier reads that: a healthy window (the same-operating-point OpenDSS
  reference of a corpus row, or a clean OPF row, with fresh noise) that does
  not alarm, plus one biased meter that does. No probe row needs phasors, so
  the request rate should be the same on both backgrounds. 1,311 probe rows
  on parents outside the train split, three kinds: a power meter at 10 to 20
  sigma, a voltage meter at 10 to 20 sigma, a voltage meter at 4.5 to 9 sigma
  (every bad meter of the training population is at least ten sigma).
- *Location.* A separate study holds waveform locations out of training.

**LLM arm.** `llm_dataset` renders the agent's decision right after the
opening WLS exactly as the DAgger pipeline does (canonical tools, compacted
model view, 4,200 to 5,300 tokens per prompt), under an evidence profile in
which no screen runs, with a triage contract paragraph in the system prompt.
The target is the truth-derived first action: request phasors
(`get_three_phase_context`) when the truth holds an HIF, an unbalance or a
harmonic source, else the context tool of the first balanced family. Two
prompt variants: `prompt_top5` is today's WLS summary (up to five residual
magnitudes of three sigma or more and up to five branch multipliers);
`prompt_top10_signed` asks for ten residuals with signs, of which the model
view's list cap keeps eight. Each has 2,035 training rows and 242 validation
rows (the parents the GNN holds out). The fine-tune is the pipeline's BC0
recipe (Gemma 4 12B in 4-bit, rank-16 LoRA, learning rate 1e-4, one pass,
best validation loss kept); a second pair of fine-tunes makes three passes
with the same settings. The adapter is read two ways:

- its greedy first action through the pipeline's own policy (`llm_score`),
  on the 463 test rows and 100 probe rows per kind and background;
- the probability of the request action, taken from the next-token
  distribution at the one token where the four tool calls part (`three`,
  `measurement`, `parameter`, `topology`), thresholded on the calibration
  split like every other score (`llm_score --probabilities`).

**Same-prompt control.** `prompt_control` parses the fields of the rendered
prompts themselves (chi-square ratio, largest residual, anomaly breadth, the
listed residuals and multipliers), and fits the gradient-boosted model on
the fine-tune's own 2,035 training rows and four-way target. It is read at
its largest class (the counterpart of a greedy first action) and at the
calibrated threshold. A gap between this control and the LLM is a gap of the
learner, not of what the prompt shows.

## 6. Stage 0 results (IEEE 14)

Brackets are 95% parent-bootstrap intervals. Source:
`output/classifier_triage_20261004/ieee14/report.md`.

**Request decision** (test split, 130 need phasors, 333 need none):

| Classifier | Recall | Unneeded requests | Time per alarm |
| --- | --- | --- | --- |
| Screen rule (reference) | 98.5% [95.9, 100] | 12.0% [8.7, 16.0] | 1.1 s |
| Gradient-boosted, screen outputs | 96.9% [93.8, 99.3] | 1.2% [0.3, 2.4] | screen plus 2 ms |
| Gradient-boosted, WLS-only study features | 98.5% [96.2, 100] | 10.5% [7.4, 13.9] | 2 ms |
| Gradient-boosted, today's prompt view | 100% | 12.9% [9.5, 16.6] | 2 ms |
| Gradient-boosted, prompt with ten signed residuals | 98.5% [96.3, 100] | 5.4% [3.0, 8.3] | 2 ms |
| Gradient-boosted, prompt with twenty signed residuals | 98.5% [96.3, 100] | 5.1% [2.8, 7.8] | 2 ms |
| Gradient-boosted, flat full vector | 97.7% [94.7, 100] | 10.2% [7.0, 13.9] | 2 ms |
| **GNN, residual-only** | **100%** | **0.9% [0.0, 2.1]** | 2 ms |
| GNN, with values | 100% | 0.9% [0.0, 2.1] | 2 ms |
| GNN, residual-only, probe rows in training | 100% | 1.5% [0.3, 3.0] | 2 ms |

The GNN requests on all 130 roots that need phasors and on 3 of the 333 that
do not (two single-meter roots and one multi-meter root). The 2 ms is the
feature build and one forward pass on a CPU, after the WLS. Test AUC by seed
is 99.88% to 99.98%. The flat full vector is no better than the WLS-only
features: the gain comes from the graph structure, not from seeing more
numbers.

**Probes on healthy windows** (request rate; no row needs phasors):

| Probe | Screen rule | GNN, OpenDSS background | GNN, OPF background | Difference |
| --- | --- | --- | --- | --- |
| Power meter, 10 to 20 sigma | 0% | 1.7% | 1.3% | 0.4% [-1.8, 2.8] |
| Voltage meter, 10 to 20 sigma | 97% | 11.8% | 9.6% | 2.3% [-3.5, 8.3] |
| Voltage meter, 4.5 to 9 sigma | 94% | 11.7% | 9.8% | 1.9% [-4.3, 7.7] |

No background effect: the GNN does not separate the classes by the simulator
that produced the window. It also does not key on the size of a voltage
residual; the prompt-view tabular models do (73% to 95% requests on the
small voltage meters). The screen requests phasors on almost every bad
voltage meter by design (rule D3), which is most of its 12%.

**Locations held out of training** (three folds over the 26 waveform
locations, three seeds each, threshold from seen locations only):

| Family | Recall, unseen locations | Recall, seen locations |
| --- | --- | --- |
| All | 97.0% [94.8, 98.9] (260/268) | 99.6% (259/260) |
| HIF | 97.3% (107/110) | 99.0% (99/100) |
| Unbalance | 91.7% [83.3, 98.3] (55/60) | 100% (60/60) |
| Harmonic | 100% (98/98) | 100% (100/100) |

Unneeded requests on the test negatives rise to 2.2% (mean over folds). The
eight misses are confident (scores 0.0002 to 0.06), five of them unbalance at
buses 2 and 3.

**First balanced family** (test rows that need no phasors):

| Classifier | Pick is a true family | Agrees with the oracle order on mixed roots |
| --- | --- | --- |
| Screen, first-round winner | 93.1% [90.2, 95.9] | 99.0% |
| Gradient-boosted, screen outputs | 98.7% [97.5, 99.7] | 99.0% |
| GNN, residual-only | 96.5% [94.4, 98.7] | 92.7% |
| Gradient-boosted, today's prompt view | 96.9% [94.3, 99.1] | 94.8% |

**HIF against same-sign flow-meter pairs.** The GNN's HIF head keeps 98% HIF
recall and flags none of the 16 same-sign pairs; the screen flags 37.5%.

**LLM arm** (fine-tuned Gemma 4 12B; same test rows). "Trees, same prompt" is
the same-prompt control of section 5. Rows are one training pass unless they
say three.

| Reading | Request AUC | Recall | Unneeded requests | Balanced pick is a true family | Four-way first action correct |
| --- | --- | --- | --- | --- | --- |
| LLM, today's prompt, greedy first action | | 75.4% [67.5, 84.7] | 11.1% [7.9, 14.7] | 66.0% [60.3, 71.8] | 68.3% |
| LLM, signed residuals, greedy first action | | 73.1% [65.0, 83.1] | 3.0% [1.2, 4.8] | 82.1% [78.1, 86.1] | 78.6% |
| LLM, today's prompt, three passes, greedy first action | | 77.7% [70.1, 86.0] | 3.9% [2.0, 6.0] | 80.5% [75.9, 85.0] | 78.8% |
| LLM, bus and branch tables, greedy first action | | 73.8% [65.9, 83.6] | 2.4% [0.9, 4.1] | 83.3% [79.4, 87.4] | 79.3% |
| LLM, bus and branch tables, three passes at half rate, greedy first action | | 83.1% [76.7, 90.1] | 7.8% [5.0, 11.0] | 76.4% [71.6, 81.4] | 78.0% |
| LLM, today's prompt, request probability at the calibrated threshold | 91.5% [88.5, 94.4] | 100% | 85.3% [81.4, 89.1] | 77.4% [72.2, 82.9] | |
| LLM, signed residuals, request probability at the calibrated threshold | 95.6% [93.5, 97.5] | 99.2% [97.5, 100] | 75.7% [71.1, 80.4] | 85.2% [81.4, 88.8] | |
| LLM, today's prompt, three passes, request probability at the calibrated threshold | 94.7% [92.5, 96.9] | 99.2% [97.5, 100] | 55.3% [50.0, 60.9] | 84.3% [80.2, 88.3] | |
| LLM, bus and branch tables, request probability at the calibrated threshold | 97.1% [95.9, 98.4] | 100% | 45.6% [40.2, 51.3] | 85.8% [82.0, 89.4] | |
| LLM, bus and branch tables, three passes at half rate, request probability at the calibrated threshold | 95.7% [93.8, 97.4] | 100% | 49.2% [43.8, 54.5] | 85.2% [81.3, 89.0] | |
| Trees, same prompt (today's), largest class | | 95.4% [91.6, 98.6] | 1.2% [0.3, 2.4] | 96.2% [93.7, 98.4] | 95.9% |
| Trees, same prompt (signed), largest class | | 96.9% [93.3, 100] | 1.2% [0.3, 2.5] | 94.7% [91.5, 97.4] | 95.0% |
| Trees, same prompt (tables), largest class | | 98.5% [96.0, 100] | 1.2% [0.3, 2.5] | 96.2% [93.4, 98.4] | |
| Trees, same prompt (today's), calibrated threshold | 99.85% [99.67, 99.97] | 100% | 13.5% [10.0, 17.2] | 96.5% [93.9, 98.7] | |
| Trees, same prompt (signed), calibrated threshold | 99.76% [99.51, 99.95] | 100% | 6.3% [3.8, 8.8] | 95.9% [92.9, 98.4] | |
| Trees, same prompt (tables), calibrated threshold | 99.9% [99.8, 100] | 100% | 3.0% [1.2, 5.0] | 97.5% [95.3, 99.4] | |
| GNN, residual-only (from above) | 99.96% [99.88, 100] | 100% | 0.9% [0.0, 2.1] | 96.5% [94.4, 98.7] | |

In the greedy and largest-class rows a request on a root that needs none
counts as a wrong pick; in the thresholded rows the pick is the most probable
balanced action whatever the request score. The calibrated threshold is the
one that reaches the screen rule's recall on the calibration split (99.3%).
For the LLM's request probability that threshold is 0.003 to 0.008, which
admits most roots. Thresholded instead at the screen rule's rate of unneeded
requests (15.9% on the calibration split), the LLM recalls 75.4% [67.2, 85.1]
with today's prompt, 90.0% [85.2, 94.7] with signed residuals and 87.7%
[82.0, 93.4] with today's prompt after three passes; the trees on the same
prompts and the GNN recall 100% there. The probability reading agrees with
the greedy one (the most probable of the four actions is the greedy decision
on 3,164 of the 3,189 rows scored both ways in these three runs) and takes
0.95 s per row.

Three passes help today's prompt and do not close the gap: unneeded requests
fall from 11.1% to 3.9% and the four-way accuracy rises from 68% to 79%, but
recall stays at 78% [70, 86]. The three-pass run with signed residuals has no
row because its training diverged (below).

Requests on the test roots that need phasors, by family:

| Family | Roots | LLM, today's prompt | LLM, signed residuals | LLM, today's prompt, three passes | LLM, tables | LLM, tables, three passes at half rate | Trees, same prompt, largest class | GNN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmonic | 50 | 50 | 50 | 50 | 50 | 50 | 50, 50 and 50 | 50 |
| Unbalance | 30 | 28 | 28 | 26 | 28 | 28 | 28, 29 and 29 | 30 |
| HIF | 25 | 17 | 17 | 18 | 17 | 21 | 24, 24 and 25 | 25 |
| HIF with a bad meter | 25 | 3 | 0 | 7 | 1 | 9 | 22, 23 and 24 | 25 |

The tables variant (one pass, standard rate; 2026-10-07) shows the bus and
branch tables of the alarm's neighbourhood in the WLS summary
(`psse_env/providers/wls_tables.py`, about 1,200 more prompt tokens). Its
validation loss fell at every evaluation (0.0549, 0.0331, 0.0212, 0.0201 at
step 509), the lowest of the one-pass runs. The tables lift the LLM's
ranking: request AUC 97.1%, the best of every LLM run, 90.0% [85.1, 95.2]
recall at the screen's rate of unneeded requests, and 45.6% unneeded at the
screen's recall instead of 76% to 85%. They do not lift its decision: the
greedy first action requests on 73.8% of the roots that need phasors and on
1 of the 25 HIF roots behind a bad meter, the same voltage-channel cue as
before. Trees on the same tables reach 98.5% recall with 1.2% unneeded at
their own decision and 3.0% unneeded at full recall: the tables carry the
information, the LLM's first action does not use it.

The last run of the leg, three passes over the tables at half the rate
(5e-5; the signed three-pass run had diverged at 1e-4), trained without a
spike (validation loss between 0.023 and 0.038 throughout, best at step 640
of 1,527). It moves the greedy decision along the same curve rather than up
it: recall 83.1% [76.7, 90.1], the highest of every LLM run (9 of 25 HIF
roots behind a bad meter, 21 of 25 HIF roots), for 7.8% unneeded requests
instead of 2.4%, and a request AUC of 95.7%, below the one-pass run's
97.1%. Eleven of its 1,063 greedy generations carry no tool call.

On the roots it misses, the LLM opens the measurement context (31 of 32
misses with today's prompt, 34 of 35 with signed residuals, 28 of 29 after
three passes). With today's prompt and one pass it never opens the parameter
context: 70 of the 75 parameter-first rows go to topology. After three passes
it does (65 of 75), and sends 43 of the 97 topology-first rows to the
parameter context instead.

What the fine-tune learned is mostly two cues. A depth-3 decision tree on
the prompt fields reproduces 94% (today's prompt) and 98% (signed) of its
request decisions; its first split is "the largest residual is a voltage
magnitude" and its second that this residual is at least about as large as
the largest branch multiplier. A voltage residual leads the list on 84 of
the 130 roots that need phasors (every harmonic root, 29 of 30 unbalance
roots, 5 of 25 HIF roots, no HIF root with a bad meter) and on 60 of the 333
that do not. On the 20 HIF roots without that cue the LLM requests on 12.
The probes show the same cue in the greedy first actions: with today's
prompt the LLM requests on 100% of the 10 to 20 sigma voltage-meter errors
and on 80% of the 4.5 to 9 sigma ones (bad power meters: 10%); with signed
residuals on 59%, 25% and 2%. Neither shows a background effect. Three
passes keep the cue (the same first split, 95% of the request decisions
reproduced; 91% and 35% on the two voltage-meter probes, 3% on bad power
meters).

Fine-tune facts: 2.4 GPU hours per pass (RTX PRO 6000). An answer is 19 to
21 tokens of which one carries the decision, so a validation loss of 0.033
per answer token is about 0.65 nats per decision, in line with the four-way
accuracies. A greedy decision takes 2.3 s (one policy step, which the agent
spends anyway). The trainer renders 19 tool schemas and the policy 18
(`run_alternative_test` is hidden under the profile); the pipeline's cells
train and evaluate with a difference of the same kind.

Validation loss per answer token (the scored adapter is the checkpoint with
the lowest one):

| Run | Steps 128, 256, 384, ... | Scored checkpoint |
| --- | --- | --- |
| Today's prompt, one pass | 0.0356, 0.0335, 0.0336, 0.0361 (step 509) | step 256 |
| Signed residuals, one pass | 0.0362, 0.0325, 0.0456, 0.0232 (step 509) | step 509 |
| Today's prompt, three passes | 0.0331, 0.0308, 0.0280, 0.0483, 0.0284, 0.0290, 0.0348, 0.0283, 0.0312, 0.0305, 0.0241, 0.0260 (step 1,527) | step 1,408 |
| Signed residuals, three passes | 2.9063, 0.0765, 0.0698, 0.0704, 0.0677, 0.0678, 0.0666, 0.0571, 0.0545, 0.0494, 0.0510, 0.0482 (step 1,527) | step 1,527 |

The three-pass run with signed residuals diverged: its training loss per
token jumped above 8 at steps 82 to 86, where the learning rate is still near
its peak (the three-pass schedule decays three times more slowly), and the
run never returned to the one-pass level. The scored adapter requests on none
of the 130 roots that need phasors, answers `get_measurement_context` on
1,004 of 1,063 rows, produces no valid tool call on 22, and its request
probability ranks the roots worse than chance (AUC 28.8%). It is a failed
training run, not evidence about the prompt; the one-pass run stays the
reference for signed residuals. Both three-pass jobs were preempted twice and
resumed from their newest checkpoints.

**Zero-shot look at IEEE 57** (the IEEE 14 weights and threshold on the 184
alarmed rows of `ieee57_v3`: 112 HIF, 46 unbalance, 26 healthy windows; its
HIFs are the normalized per-unit sweep and it has no balanced-fault roots):

| Classifier | Recall | Requests on healthy windows |
| --- | --- | --- |
| Screen rule | 63.9% [56.2, 71.7] | 3/26 |
| GNN, residual-only | 50.0% [42.0, 58.1] (AUC 84.2%) | 0/26 |
| GNN, with values | 31.0% [24.1, 38.6] | 0/26 |

## 7. Reading

1. On IEEE 14 the exhaustive screen is not needed for the request decision.
   A residual-only GNN reaches the screen's recall with about a tenth of its
   unneeded requests, at 2 ms.
2. The result does not rest on the cues that could be probed: operating-point
   values add nothing, there is no simulator-background effect, and small
   voltage-meter errors outside the training range are handled like large
   ones.
3. "No retraining across networks" is not supported yet. Within IEEE 14 the
   GNN loses three points of recall at unseen locations (eight for
   unbalance), and the first zero-shot look at IEEE 57 gives half the recall.
   The residual-only view transfers better than the values view (50% against
   31%), which supports the feature choice, but training on more than one
   network (or leave-one-network-out) is needed before the claim can be
   tested fairly.
4. Under G1 a confident miss is not recovered by the gate. The two fallbacks
   of section 2 are what keeps such a root reachable, at the price of extra
   steps.
5. As trained here, the LLM is not a triage classifier. With the pipeline's
   fine-tuning recipe its first action requests on about three quarters of
   the roots that need phasors (the GNN: all of them), and its request
   probability ranks the roots with an AUC of 92% to 96% (the GNN: 99.96%). A
   threshold does not rescue it: to reach the screen's recall the LLM has to
   request on 55% to 85% of the roots that need nothing. Three training
   passes instead of one make it more precise (unneeded requests 11.1% to
   3.9% with today's prompt) and leave recall at 78% [70, 86]. The recipe is
   also fragile on this task: one of the two three-pass runs diverged.
6. The prompt is not what limits it. Trees fitted on the fields the prompt
   shows, on the same rows and targets, reach 95% to 97% recall with 1.2%
   unneeded requests at their own decision, and full recall at the calibrated
   threshold. The fine-tune extracted mostly a voltage-channel cue (a
   voltage residual leading the list and not dwarfed by the branch
   multipliers), which finds harmonic and unbalance roots and misses an HIF
   behind a bad meter.
7. Signed residuals and the bus and branch tables help every reader of the
   prompt's ranking: the trees' unneeded requests at full recall fall from
   13.5% (today's prompt) to 6.3% (signed) and 3.0% (tables), the one-pass
   LLM's AUC rises from 91.5% to 95.6% and 97.1%. None of them moves the
   LLM's greedy decision off the voltage cue (recall 73% to 75% in every
   one-pass run). The model view caps a list at eight entries; the cap has to
   be raised for the agent's WLS summary to show ten signed residuals (the
   tables need no cap change, `WLS_SUMMARY_TABLE_KEYS`).
8. For the two arms this means: on IEEE 14 the request gate should be the
   GNN's score, with the LLM acting on its report. An LLM that classifies by
   itself needs more than this recipe, and three passes did not supply it.
   The comparison that stays meaningful for the LLM is the agent with the
   classifier's report against the agent with the screen's report, not the
   LLM as the classifier. Stage 1 (section 10) shows the classifier
   profile gives the rule expert the same roots as the screen profile at
   under half the wall time; the DAgger cell on that profile is the next
   step for the GNN arm.

## 8. Open

- **LLM as the classifier.** Closed for IEEE 14 (2026-10-07). Six
  fine-tunes over three prompt forms, one and three passes, two learning
  rates (one run diverged): the best greedy decision is 83.1% recall at 7.8% unneeded
  requests, the best ranking an AUC of 97.1%, against 98.5% / 1.2% for trees
  on the same tables and 100% / 0.9% for the GNN. An LLM-arm DAgger cell
  with the LLM as the classifier is not justified by the offline evidence;
  the LLM's place is acting on the GNN's report (the cell of section 11).
  Not tried: more training roots, a larger model, or a reasoning trace
  before the decision.
- **Stage 2 (GNN arm).** The DAgger cell on `classifier_gated_diagnostics`:
  add the profile to `pipeline.env` (and the string assertions in
  `research/test_hpc_full_pipeline.py`), the triage export as the cell's
  classifier, the baseline expert as teacher (the ledger variants read the
  screen), and the teacher fix for the voltage-meter inconsistency first.
- **Second decision on mixed roots.** A first action understates a closed
  loop on a root with a bad meter and an HIF: after the meter is corrected
  the agent decides again on the HIF that remains. This is not measured
  offline (the test split holds roots only); on the HIF-only roots the
  one-pass LLM requests on 17 of 25, so a second decision would recover part
  of the gap at best.
- **List cap of the model view.** Eight entries; ten signed residuals need a
  higher cap for the WLS summary.
- **Stage 1.** The evidence profile without the screen (gate G1, the two
  fallbacks, the triage report in the observation for the GNN arm), then the
  rule expert end to end with each triage source.
- **Teacher.** Done 2026-10-06 (commit 1003f7f): under the suspicion
  profile, phasors that came back balanced on a voltage-meter suspicion now
  lead to the spectra before the meter is edited, whether or not the
  screen's hypotheses explained the alarm. On the 160 development roots the
  Step-5 teacher stays at 160/160; the two harmonic roots it used to
  spend 13 and 15 steps on (testing the screen's bus-9 voltage-meter
  hypothesis twice) take 6, the true voltage-meter root and one multi-meter
  root take one step more, mean steps 7.9 to 7.8, spectra 16 to 18. The
  classifier profile's expert had this order from the start.
- **Prompt text.** The suspicion-gated paragraph of the system prompt still
  describes the 2026-09-27 rule (phasors only on an HIF suspicion); it is
  replaced with the new profile.
- **Corpus format cue.** Topology roots are the only ones on the node-breaker
  operator model (exact zero-injection rows, different sigmas). Neutralized
  in the features here; the lasting fix is one operator model for all roots.

## 10. Stage 1: the GNN arm's profile (built 2026-10-06)

`classifier_gated_diagnostics` (`psse_env/evidence_profile.py`) is the
suspicion-gated contract with the balanced screen taken out and the triage
classifier put in its place. What changed, and where:

- **The report.** The WLS provider loads the exported classifier once
  (`research/classifier_triage/runtime.py`; the tracked export is
  `psse_env/oracle/models/triage_gnn_ieee14_20261004`, five seeds in half
  precision, 6.9 MB, threshold 0.2782 from the benchmark's calibration) and
  after every solve writes `triage` on the WLS ledger entry beside the
  detection metrics: method and model id, status, request score and
  threshold, `request_admitted`, the first balanced family with its three
  scores, the six family scores as a diagnostic, and the state binding. It
  reads the operator's current case (a corrected model has its own
  parameters) and the solve; nothing else. A classifier failure is an
  unavailable report that admits nothing; the solve stands.
- **The gate (G1).** `required_suspicion` asks the same families as under the
  suspicion profile; `current_suspicion` answers them from the report
  instead of the screen: `phasor` is an admitted request on the current
  bound WLS, or the fallback `balanced_route_failed`: a balanced correction
  bound to the active state was rejected by verification (an executor
  failure tests nothing), or every balanced context fetched on the state
  offered no correction. `hif` is what the phasors showed (the NLM's
  `hif_suspected`), `harmonic` phasors that came back balanced, as before.
  A candidate's verification solve carries the report for the candidate
  state. Nothing is minted into the signatures.
- **The expert.** One screening stage (`classifier_screening_proposals`):
  an admitted request acquires the phasors, the NLM tests them once, balanced
  phasors open the spectra. Without an admitted request the balanced ladder
  runs with the report's first family ahead of the others
  (`_triage_first_order`), inside what the Lagrangian dominance tags allow:
  a dominant residual still suppresses the branch routes, a rule that is
  right on meter-against-branch 99.4% of the time on the test split. When
  the ladder is exhausted, `unexplained_acquisition_proposals` opens the
  phasors through the fallback and the handoff follows as under the
  suspicion profile.
- **Shared contract.** The data and diagnostics contract of the suspicion
  profile applies to both (`is_suspicion_profile`): true-state phasors and
  clean spectra on every root at the uniform PMU sigma, the NLM and the HIF
  estimator from the snapshot phasors, balanced split-line conditioning of an
  accepted HIF, the HIF acquisition block and scan window dropped from the
  metadata, the multi-scan estimator hidden. The learned ranker's deferral
  and the D3 voltage-meter hold stay suspicion-only (they read the screen).
- **Model view and prompt.** `triage` is a history metric and a context
  detail key, so it survives compaction in the WLS tool output and the
  ledger; the system prompt gets its own paragraph
  (`CLASSIFIER_GATED_PROMPT_PARAGRAPH`), which states the gate as built.
- **Episode check.** `research/classifier_triage/expert_e2e.py` rolls the
  rule expert out on the pipeline's development suite under a chosen
  profile and reports, per family, success, acquisitions, unnecessary
  acquisitions, false commits, episode length, and the first triage report
  (its closed-loop recall and unneeded rate beside the offline figures).

Tests: `psse_env/oracle/test_classifier_gated_routing.py` (profile, gate,
fallbacks, expert routing, verification), `research/classifier_triage/tests/test_runtime.py`
(export, load, report), `psse_env/providers/test_wls_tables.py` (the tables
and their model view).

**Result on the 160 development roots** (the pipeline's `development.json`
of the ranked cell: 16 each of harmonic, HIF, measurement, parameter,
topology, unbalance and measurement+parameter, 12 each of multi-meter and
measurement+topology, 8 each of measurement+HIF, healthy and
telemetry-only; `research/classifier_triage/expert_e2e.py`, seed 20261006,
40 steps, one CPU):

| Teacher | Success | Phasors acquired (not needed) | Spectra (not needed) | Healthy components touched | Mean steps | Wall time |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier profile, baseline expert | 159/160 | 59 (3) | 18 (2) | 5 | 8.0 | 4.1 min |
| Screen profile, baseline expert | 159/160 | 59 (3) | 17 (1) | 5 | 8.6 | 10.0 min |
| Screen profile, ledger expert with the ranker (the Step-5 teacher) | 160/160 | 58 (2) | 16 (0) | 4 | 7.9 | 9.4 min |

"Not needed" counts phasors on roots other than HIF, HIF with a meter and
unbalance, less the 16 harmonic roots, whose phasors are the contract's
first tier before the spectra in every teacher. The classifier's first
report admitted phasors on all 40 roots that need them and on 1 of the 104
that do not (a multi-meter root at score 0.31); the other two acquisitions
it did not need came later in their episodes, one through the fallback
after a rejected correction and one from the report on a corrected child
state.

With the same expert, the classifier profile matches the screen profile
root for root: the same 159 successes, the same failing root, 2.4 times
less wall time. The one root the Step-5 teacher solves and both baseline
teachers miss is a parameter root whose true line ranks second in the
multiplier ranking (dominance ratio 1.02); the screen's hypothesis ledger
names the right line, the parameter context's own ranking names the wrong
one, which the baseline expert corrects and commits before the episode
goes astray. That is localization inside the balanced ladder, which the
classifier does not do (by design, section 3), not the request decision.
Carrying the ledger's candidate ranking without the screen's refits is the
open item for the balanced ladder.

Known gaps: a root where the classifier does not admit phasors and the
balanced ladder finds nothing to try ends in an operator handoff rather
than a phasor request (the fallback needs a fetched context that offered
nothing, or a rejected correction); the parameter ranking's misranked
stratum (above); the pipeline cell (`pipeline.env`) does not yet list the
profile; the GNN is the IEEE 14 model, so the profile is usable on IEEE 14
roots only until a multi-network model exists.

## 11. Stage 2: the DAgger cell on the classifier profile (launched 2026-10-07)

Cell `/scratch/yx3882/research_full_pipeline_20261007_classifier` on torch,
source 1003f7f, overrides
`research/hpc/full_pipeline_20260907/overrides/classifier_gated_20261007.env`
(`EVIDENCE_PROFILE=classifier_gated_diagnostics`, `EXPERT_VARIANT=baseline`,
`HIF_SIGNATURE_MODE=discovered`, nothing reused from an earlier cell).
Corpora, plans (548 D0 roots, 160 development roots, 122 roots per round),
seeds, the 40-step budget and the training recipe are those of the
2026-10-01 ranked cell; the teacher differs (baseline expert reading the
triage report instead of the ledger expert reading the screen) and so does
the profile, so D0, BC0 and both rounds are regenerated. The deploy's
dry-run prerequisites loaded the triage export
(`triage_gnn:5d6df27b1aefc3fb`, five seeds, threshold 0.2782) and the
corpora' PMU sigma. Chain submitted 02:01:56 UTC: d0 19314399, bc0 19314401,
r1c 19314404, r1t 19314405, r1e 19314407, r2c 19314408, r2t 19314409,
r2e 19314410.

Pre-flight: a 24-root expert aggregate (two roots per family) with the
cell's exact settings ran locally under the profile without a collector or
training-decision failure (193 raw rows); the Stage 1 check (section 10) is
the same teacher on the development roots.

What the cell measures against the 2026-10-01 ranked cell (expert 160, BC0
152, R1 158, R2 156 of 160 on its own development draw): the student's
success and the number of phasor and spectra requests it makes, with the
expert arm of every evaluation being the baseline expert on the classifier
profile. Status:
`MSYS_NO_PATHCONV=1 wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20261007_classifier/status_pipeline.sh`.

## 9. Reproduce

```bash
python -m research.classifier_triage.benchmark --output-dir output/classifier_triage_20261004/ieee14 --seeds 5
```

```bash
python -m research.classifier_triage.location_study --output-dir output/classifier_triage_20261004/ieee14_location --seeds 3
```

```bash
python -m research.classifier_triage.transfer_look --benchmark-dir output/classifier_triage_20261004/ieee14
```

The benchmark takes about ten minutes with a local GPU (probe build three
minutes, fifteen GNN trainings six). `--reuse-gnn` reloads saved GNN scores
and scores new probe rows with the saved checkpoints.

LLM leg. The prompt builds run locally:

```bash
PSSE_LOCAL_DIAGNOSTIC_BUILD=1 python -m research.classifier_triage.llm_dataset --output-dir output/classifier_triage_20261004/llm --variant prompt_top10_signed
```

Training and scoring run on the cluster. The deploy script uploads the
source and the datasets and submits one job per variant (`llm_triage.sbatch`:
fine-tune, greedy first actions, first-action probabilities; `EPOCHS` and
`RUN_TAG` in the job's environment give a longer run its own output
directory):

```bash
bash research/hpc/classifier_triage_20261004/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

The three-pass runs were submitted from the cluster work directory, one per
variant (results under `out/<variant>_e3`):

```bash
sbatch --export=ALL,VARIANT=prompt_top5,EPOCHS=3,RUN_TAG=_e3 --job-name=triage-prompt_top5-e3 source/research/hpc/classifier_triage_20261004/llm_triage.sbatch
```

The results (`out/<run>/scores.json` and `probabilities.json` in the cluster
work directory) are copied to
`output/classifier_triage_20261004/llm_results/<run>/`:

```bash
bash research/hpc/classifier_triage_20261004/fetch_results.sh prompt_top5 prompt_top10_signed
```

They enter the tables with the same-prompt control (each option repeats per
variant):

```bash
python -m research.classifier_triage.benchmark --output-dir output/classifier_triage_20261004/ieee14 --reuse-gnn --prompt-control prompt_top5=output/classifier_triage_20261004/llm/prompt_top5 --llm-scores llm_prompt_top5=output/classifier_triage_20261004/llm_results/prompt_top5/scores.json --llm-probabilities llm_prompt_top5_probability=output/classifier_triage_20261004/llm_results/prompt_top5/probabilities.json
```
