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
LLM is the classifier, so its own decision is the gate; offline it is scored
at that decision, and the policy wrapper can threshold the probability of the
request action when a tunable operating point is needed.

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
| 0 | Offline benchmark on the IEEE 14 study rows: screen rule, gradient-boosted models, GNN, LLM | CPU and one local GPU; the LLM leg needs a cluster fine-tune | GNN and tabular legs done (section 6); LLM leg running on the cluster (section 8) |
| 1 | New evidence profile without the screen; rule expert driven by each triage source on the 160 development roots | CPU | not started |
| 2 | One DAgger cell per arm | about a day each | not started |
| later | IEEE 57 and 118 roots of every family; leave-one-network-out | generation plus training | not started |

## 5. Stage 0 design

Code: `research/classifier_triage/` (`data`, `features`, `model`,
`train_gnn`, `benchmark`, `location_study`, `transfer_look`), tests in
`research/classifier_triage/tests/`.

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
5. For the LLM arm, what the prompt shows bounds the result. A tabular model
   on today's prompt view sits at the screen's level (12.9% unneeded); with
   ten signed residuals it halves that (5.4%). The agent's WLS summary should
   list ten residuals with signs in the LLM arm.

## 8. Open

- **LLM leg of stage 0.** Running on the cluster since 2026-10-04 05:45 UTC:
  jobs 19142100 (`prompt_top5`) and 19142190 (`prompt_top10_signed`), work
  directory `/scratch/yx3882/classifier_triage_20261004`, source eb4e5be,
  submitted with `research/hpc/classifier_triage_20261004/deploy_from_windows.sh`.
  `llm_dataset` renders the agent's decision after the opening
  WLS exactly as the DAgger pipeline does (canonical tools, compacted model
  view), with no screen report and a triage contract paragraph, and the
  truth-derived first action as the target. Two prompt variants are built
  under `output/classifier_triage_20261004/llm/`: `prompt_top5` (today's WLS
  summary) and `prompt_top10_signed`; each has 2,035 training rows, 242
  validation rows (the parents the GNN holds out) and 2,232 scored prompts
  (calibration, test and probes), and passes the trainer's split and protocol
  gates. `llm_score` runs the fine-tuned adapter through the pipeline's own
  policy on the scored prompts and `benchmark --llm-scores NAME=PATH` adds
  its first actions to the tables. The trainer's tokenizer audit passed on
  the cluster for both variants (4,200 to 5,300 tokens per prompt). Each job
  trains one pass with the BC0 recipe (about three GPU hours by the BC0
  rate), then scores the 463 test rows and 100 probe rows per kind and
  background (an LLM's first action needs no fitted threshold, so the
  calibration rows are not scored).
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

LLM leg (the two builds run locally; training and scoring need the cluster):

```bash
PSSE_LOCAL_DIAGNOSTIC_BUILD=1 python -m research.classifier_triage.llm_dataset --output-dir output/classifier_triage_20261004/llm --variant prompt_top10_signed
```

```bash
python -m research.classifier_triage.llm_score --adapter OUT/lora --score DATA/score.jsonl --output OUT/scores.json
```
