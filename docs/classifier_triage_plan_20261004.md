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
| 0 | Offline benchmark on the IEEE 14 study rows: screen rule, gradient-boosted models, GNN, LLM | CPU and one local GPU; the LLM leg needs a cluster fine-tune | done with one training pass per LLM variant (section 6); three-pass fine-tunes running (section 8) |
| 1 | New evidence profile without the screen; rule expert driven by each triage source on the 160 development roots | CPU | not started |
| 2 | One DAgger cell per arm | about a day each | not started |
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
best validation loss kept). The adapter is read two ways:

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

**LLM arm** (fine-tuned Gemma 4 12B, one pass; same test rows). "Trees, same
prompt" is the same-prompt control of section 5.

| Reading | Request AUC | Recall | Unneeded requests | Balanced pick is a true family | Four-way first action correct |
| --- | --- | --- | --- | --- | --- |
| LLM, today's prompt, greedy first action | | 75.4% [67.5, 84.7] | 11.1% [7.9, 14.7] | 66.0% [60.3, 71.8] | 68.3% |
| LLM, signed residuals, greedy first action | | 73.1% [65.0, 83.1] | 3.0% [1.2, 4.8] | 82.1% [78.1, 86.1] | 78.6% |
| LLM, today's prompt, request probability at the calibrated threshold | 91.5% [88.5, 94.4] | 100% | 85.3% [81.4, 89.1] | 77.4% [72.2, 82.9] | |
| LLM, signed residuals, request probability at the calibrated threshold | 95.6% [93.5, 97.5] | 99.2% [97.5, 100] | 75.7% [71.1, 80.4] | 85.2% [81.4, 88.8] | |
| Trees, same prompt (today's), largest class | | 95.4% [91.6, 98.6] | 1.2% [0.3, 2.4] | 96.2% [93.7, 98.4] | 95.9% |
| Trees, same prompt (signed), largest class | | 96.9% [93.3, 100] | 1.2% [0.3, 2.5] | 94.7% [91.5, 97.4] | 95.0% |
| Trees, same prompt (today's), calibrated threshold | 99.85% [99.67, 99.97] | 100% | 13.5% [10.0, 17.2] | 96.5% [93.9, 98.7] | |
| Trees, same prompt (signed), calibrated threshold | 99.76% [99.51, 99.95] | 100% | 6.3% [3.8, 8.8] | 95.9% [92.9, 98.4] | |
| GNN, residual-only (from above) | 99.96% [99.88, 100] | 100% | 0.9% [0.0, 2.1] | 96.5% [94.4, 98.7] | |

In the greedy and largest-class rows a request on a root that needs none
counts as a wrong pick; in the thresholded rows the pick is the most probable
balanced action whatever the request score. The calibrated threshold is the
one that reaches the screen rule's recall on the calibration split (99.3%).
For the LLM's request probability that threshold is 0.003 to 0.004, which
admits most roots. Thresholded instead at the screen rule's rate of unneeded
requests (15.9% on the calibration split), the LLM recalls 75.4% [67.2, 85.1]
with today's prompt and 90.0% [85.2, 94.7] with signed residuals; the trees
on the same prompts and the GNN recall 100% there. The probability reading
agrees with the greedy one (the most probable of the four actions is the
greedy decision on 2,107 of the 2,126 rows scored both ways) and takes 0.95 s
per row.

Requests on the test roots that need phasors, by family:

| Family | Roots | LLM, today's prompt | LLM, signed residuals | Trees, same prompt, largest class | GNN |
| --- | --- | --- | --- | --- | --- |
| Harmonic | 50 | 50 | 50 | 50 and 50 | 50 |
| Unbalance | 30 | 28 | 28 | 28 and 29 | 30 |
| HIF | 25 | 17 | 17 | 24 and 24 | 25 |
| HIF with a bad meter | 25 | 3 | 0 | 22 and 23 | 25 |

On the roots it misses, the LLM opens the measurement context (31 of 32
misses with today's prompt, 34 of 35 with signed residuals). With today's
prompt it never opens the parameter context: 70 of the 75 parameter-first
rows go to topology.

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
residuals on 59%, 25% and 2%. Neither shows a background effect.

Fine-tune facts: 2.4 GPU hours per variant (RTX PRO 6000); validation loss
per answer token 0.0335 with today's prompt (best checkpoint at step 256 of
509) and 0.0232 with signed residuals (at the last step, and still moving:
0.0456 at step 384). An answer is 19 to 21 tokens of which one carries the
decision, so these are about 0.65 and 0.45 nats per decision, in line with
the four-way accuracies. A greedy decision takes 2.3 s (one policy step,
which the agent spends anyway). The trainer renders 19 tool schemas and the
policy 18 (`run_alternative_test` is hidden under the profile); the
pipeline's cells train and evaluate with a difference of the same kind.

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
5. As trained here, the LLM is not a triage classifier. After one pass of the
   pipeline's fine-tuning recipe its first action requests on about three
   quarters of the roots that need phasors (the GNN: all of them), and its
   request probability ranks the roots with an AUC of 92% to 96% (the GNN:
   99.96%). A threshold does not rescue it: to reach the screen's recall the
   LLM has to request on 76% to 85% of the roots that need nothing.
6. The prompt is not what limits it. Trees fitted on the fields the prompt
   shows, on the same rows and targets, reach 95% to 97% recall with 1.2%
   unneeded requests at their own decision, and full recall at the calibrated
   threshold. The fine-tune extracted mostly a voltage-channel cue (a
   voltage residual leading the list and not dwarfed by the branch
   multipliers), which finds harmonic and unbalance roots and misses an HIF
   behind a bad meter.
7. Signed residuals help every reader of the prompt: the trees' unneeded
   requests at full recall fall from 13.5% to 6.3%, the LLM's AUC rises from
   91.5% to 95.6% and its recall at the screen's rate of unneeded requests
   from 75% to 90%. The model view caps a list at eight entries; the cap has
   to be raised for the agent's WLS summary to show ten.
8. For the two arms this means: on IEEE 14 the request gate should be the
   GNN's score, with the LLM acting on its report. An LLM that classifies by
   itself needs more than this recipe: three training passes are running
   (section 8). If they do not close the gap, the comparison that stays
   meaningful for the LLM is the agent with the classifier's report against
   the agent with the screen's report, not the LLM as the classifier.

## 8. Open

- **Longer LLM fine-tunes.** One pass is short for this task (the
  `prompt_top5` run kept its checkpoint of step 256 of 509; the
  `prompt_top10_signed` run was still improving at its last step). Three
  passes of both variants are running on the cluster since 2026-10-04 09:33
  UTC: jobs 19149103 (`prompt_top10_signed`) and 19149112 (`prompt_top5`),
  work directory `/scratch/yx3882/classifier_triage_20261004`, outputs
  `out/<variant>_e3`, source 200f936, 1,527 optimizer steps at about 15 s
  each, then both scorings (about nine hours in all). Fetch with
  `research/hpc/classifier_triage_20261004/fetch_results.sh prompt_top5_e3
  prompt_top10_signed_e3` and add them to the benchmark command of section 9.
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
- **Teacher.** Spectra before a voltage-meter edit when phasors came back
  balanced (the inconsistency that cost R2 three roots on 2026-10-02) before
  any new cell.
- **Prompt text.** The suspicion-gated paragraph of the system prompt still
  describes the 2026-09-27 rule (phasors only on an HIF suspicion); it is
  replaced with the new profile.
- **Corpus format cue.** Topology roots are the only ones on the node-breaker
  operator model (exact zero-injection rows, different sigmas). Neutralized
  in the features here; the lasting fix is one operator model for all roots.

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
