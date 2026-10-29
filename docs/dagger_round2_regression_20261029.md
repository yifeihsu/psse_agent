# Why DAgger round 2 can score below round 1, and the 2026-10-29 round recipe

Research-only analysis of the full research pipeline
(`research/hpc/full_pipeline_20260907`: BC0 from the expert aggregate, then
DAgger rounds R1 and R2 on disjoint 122-root draws, paired evaluation on the
same development roots). Nothing here is release evidence. The release
DAgger-1 path in `psse_env/dagger` is a different, single-round design and
is not changed by this note.

## 1. What the cells recorded

Truth-audited task success on the cell's development roots (160 unless
noted), with the expert rolled out on the same roots as the ceiling:

| Cell | BC0 | R1 | R2 | Expert | R2 against R1 |
| --- | ---: | ---: | ---: | ---: | --- |
| 2026-09-07 first run | 131 | 117 | 118 | | topology 16 to 0 to 0: context-fetch labels at the learner's states were quarantined by the rank-one proof and nothing stopped a masking meter edit (fixed by the ambiguity design, `docs/ambiguity_rerun_20260908.md`) |
| 2026-09-08 ambiguity rerun | 144 | 147 | 146 | 147 | R2 recovers one HIF root and loses two measurement+parameter roots to the commit-rollback loop; false commits 18 to 31, invalid actions 27 to 39; dominant parameter stratum BC0 22/25, R1 25/25, R2 23/25 |
| 2026-09-09 ten sigma | 125 | 146 (147 re-evaluated) | 147 | 147 | parity; one multi-meter root flips between two evaluations of the same adapter (greedy decoding on different GPU nodes) |
| 2026-09-24 WLS-gated | 156 | 157 | 157 | 158 | parity |
| 2026-10-01 ranked screen cell | 152 | 158 | 156 | 160 | R2 gains the topology root R1 missed and loses three roots, all at the step after balanced phasors on a voltage-meter suspicion (`docs/hypothesis_ranking_step5_20261001.md`) |
| 2026-10-07 classifier-gated cell | 153 | 159 | 159 | 159 | parity with fewer invalid actions and false commits |
| 2026-10-22 conventional-only cell (the paper's; 192 fresh roots) | 189 | 191 | 190 | 192 | R2 loses one breaker root and has 9 invalid-action and 2 loop episodes where R1 has none (section 4) |

So R2 scored below R1 in three of the seven cells, by one or two roots, and
never above it by more than a root. The regressions have a recognisable
signature rather than random loss: in the 2026-09-08 cell R2 re-acquired a
failure mode that R1 had removed from BC0 (the parameter commit-rollback
loop; BC0 22/25, R1 25/25, R2 23/25 on the dominant stratum), in the paper's
cell R2's violations fall on the multi-error cohorts where BC0's did and R1's
did not (section 4), and in the 2026-10-01 cell R2 flipped at a decision
where the teacher's own label is not a function of the observation. All
three are what the round recipe below predicts.

## 2. How the rounds were built (through 2026-10-28)

`pipeline.env`, `stage_collect.sbatch`, `stage_train.sbatch` and
`scripts/run_dagger_research.py`:

1. Round k's student rolls out on 122 fresh roots at beta 0.25 (the expert
   executes a quarter of the steps; it labels every visited state). Rows
   at learner-visited states that pass the rank-one proof and the offline
   truth audit are the round's safe rows D_k (about 530 to 580 of 122
   roots; 47 to 55 learner-state labels are quarantined per round).
2. The training pass is D_k plus an equal-size **uniform sample** of the
   round's pool (`D1_SHARE=0.5`). Round 1 pools the expert aggregate D0;
   round 2 pools D0 plus D1 (`stage_collect.sbatch` builds
   `out/r2/pool/`).
3. The candidate is the collecting student continued for one pass at
   3e-5 (linear schedule and optimizer state restarted each round), and
   the **final** checkpoint is kept.

### 2.1 Round 2 did not train on the DAgger aggregate

DAgger's guarantee (Ross, Gordon and Bagnell, 2011) is for a policy trained
on the union of every round's data, D0 u D1 u ... u Dk. That union is what
keeps a correction learned at round k's failure states in the training
pass of round k+1. In the pool recipe D1 competes with D0 for the sampled
replay slots. With a 3,300 to 3,700-row D0 training view and 540 to 580 D1
rows, D1 is 13 to 15 percent of the pool, so about 70 to 80 of round 1's
rows reached round 2's 1,070-row pass; the remaining 85 to 87 percent were
absent. Round 2's own rows do not replace them: they come from different
roots, and 91 to 97 percent of them sit at states where R1 already agreed
with the teacher. The measurement+parameter commit-rollback loop of the
2026-09-08 cell is exactly a correction R1 had learned from D1 and R2 was
trained almost without.

### 2.2 The candidate is a small, unselected perturbation of a policy at the ceiling

R2 is R1 plus one pass over about 1,070 rows, half of which carry little
gradient (states R1 already handles), half a fresh random D0 sample, with
the learning-rate schedule and Adam state restarted; no checkpoint or
round selection is applied (BC0 selects by validation loss, which the
2026-09-09 write-up showed is not aligned with task success, so no round
does). A policy already at the expert's ceiling on the development roots
(147 of 152 reachable, 158 of 160) can only lose roots from such a
perturbation, and the evaluation itself flips about one root between runs
of the same adapter. Differences of one or two roots between R1 and R2 are
therefore not evidence about the data; the paired per-root record (gained
and lost roots, their failure modes) is.

### 2.3 The teacher's label was not a function of the observation at one state

In the 2026-10-01 cell the ledger teacher edits the voltage meter when the
balanced screen explained the alarm and asks for spectra when it did not;
the policy observation does not carry that distinction. The aggregate then
holds conflicting targets for indistinguishable observations, the student
is fitted to an average, and which label it follows flips from round to
round. The teacher was changed (spectra before a voltage-meter edit) and the
2026-10-07 cell, which held R1's 159 in round 2, ran under a consistent
teacher. The mixture now carries an exact label-conflict audit so this
class of defect is visible before a round is trained.

### 2.4 Contributing, not round-specific

The quarantine removes labels at the states the student needs most (about
one learner-state label in ten), and a later round's fresh 122-root draw
yields few new hard states once disagreement is a few percent. Neither
causes R2 < R1 on its own, but both make a round's new information small
relative to the perturbation of 2.2.

## 3. The 2026-10-29 round recipe

Implemented in `scripts/run_dagger_research.py` and the cell scripts;
`research/hpc/full_pipeline_20260907/README.md` section "Round recipe
revision (2026-10-29)" has the operational details.

- **Cumulative aggregate** (`ROUND_AGGREGATE=cumulative`, the default; the
  runner's repeatable `--prior-onpolicy-dir`). Round k's on-policy side is
  D_k plus every earlier round's safe rows in full, re-rendered under the
  current canonical view like the D0 replay and bound by content hash in
  the run configuration; D0 is sampled so the on-policy side is
  `D1_SHARE` of the pass. Earlier rounds' roots are protected like D0
  roots: the new round neither rolls out on them nor exports rows from
  them. The mixture report records `prior_onpolicy_selected`,
  `prior_onpolicy_retained_fraction` (1.0) and the rows per earlier round.
  For a 122-root round 2 the pass grows from about 1,070 to about 2,200
  rows (about 550 optimizer steps, roughly five hours at the measured 2.5
  hours per 285 steps).
- **Fixed initialization** (`ROUND_INIT=bc0`, the default). Every round
  trains from the BC0 adapter on its aggregate, DAgger's rule, so rounds
  differ only by their data and a round's drift does not compound on the
  previous one. The round's student still collects on-policy, and the
  paired evaluation still reads the candidate against that student.
- **Exact label-conflict audit** on every mixture
  (`label_conflicts` in `mixture_report.json`, mirrored in the
  `collection.done` receipt): canonical observations rendered with more
  than one semantic target, with example ids. A nonzero count is a teacher
  or observation defect to fix before training; conflicting labels cannot
  be learned.
- The 2026-09-07 recipe stays reproducible with
  `ROUND_AGGREGATE=pool ROUND_INIT=previous`.

## 4. The paper's cell: the conventional-only confirmation (2026-10-22)

`docs/conventional_only_cell_20261022.md` is the cell whose tables the PES GM
2027 manuscript reads (Section VI): the seven conventional families only,
`scada_only`, the `baseline_compare` teacher, a fresh 192-root test set, the
same round recipe as above (pool mixture, student continued, final
checkpoint).

| | Expert | BC0 | R1 | R2 |
| --- | ---: | ---: | ---: | ---: |
| Test set, 192 fresh roots | 192 | 189 | 191 | 190 |
| Episodes with an invalid action (invalid actions) | 0 (0) | 19 (39) | 0 (0) | 9 (15) |
| Episodes with a false commit / loop episodes | 0 / 0 | 3 / 2 | 0 / 0 | 0 / 2 |
| Mean actions per episode | 9.87 | 10.38 | 9.85 | 9.93 |
| Hard-draw stress roots, 193 (process-violation episodes) | 185 (0) | 182 (22) | 183 (1) | 182 (5) |
| Validation roots, 96 (process-violation episodes) | 96 (0) | 96 (6) | 96 (1) | 96 (4) |

The round selection on the validation roots picks R1. The success
differences between R1 and R2 are within the paired noise (one discordant
root each way, p = 1); the difference in process violations on the test set
(9 episodes against 0, p = 0.004) is not.

**The arithmetic of round 2 in this cell.** D0's training view has 2,747
rows. Round 1 collected 437 safe rows with BC0 as the student (3.2% tool
disagreement) and trained R1 for 219 steps on 437 + 437. Round 2 collected
432 safe rows with R1 as the student (1.6% tool disagreement, no invalid
student action during collection) and trained R2 for 216 steps on 432 D2
rows plus 432 rows drawn uniformly from the 3,184-row pool of D0 and D1. The
expected number of round-1 rows in that draw is 432 x 437 / 3,184 = 59, so
about 86 percent of D1 was absent from R2's pass, and the pass was roughly
43 percent D0 rows R1 had not been trained on, 50 percent D2 rows at states
R1 already handled like the teacher, and 7 percent D1. To count it exactly
on the cluster (round-2 rows carry the `d1` mixture tag; round-1 rows that
survived the pool draw carry `d0` with a `dagger_iter1_` example id):

```bash
cd /scratch/yx3882/research_full_pipeline_20261022_conventional
python3 - <<'EOF'
import json
rows = [json.loads(l) for l in open("out/r2/collection/round1.train.jsonl") if l.strip()]
d1 = [r for r in rows if r.get("research_mixture_source") == "d0"
      and str(r.get("example_id", "")).startswith("dagger_iter1_")]
print(len(rows), "rows in the round-2 pass;", len(d1), "of them round-1 rows")
EOF
```

**What D1 taught and what R2 lost.** The 437 round-1 rows are 190 context
requests, 88 commits, 88 solves, 47 handoffs, 19 corrections and one
comparison, collected at the states BC0 reached on the round-1 roots: the
process discipline BC0 lacked on the multi-error roots, where all 19 of its
violation episodes fell (correction, commit and rollback cycles on
meter + R/X, repeated context requests and meter edits on meter + breaker).
R1 learned it and has no violation on any set. R2's nine violation episodes
are four multi-meter roots, three meter + breaker roots (a second
`correct_topology` before verifying the first), one meter + R/X root and the
breaker root it lost (a context request, then `correct_topology` again with
its candidate open, both refused): eight of the nine on the same multi-error
cohorts as BC0's nineteen, and all of the kind D1 corrected. That is the
pool recipe's signature, a partial return of BC0's behaviour where D1's rows
were dropped, rather than a new failure class. The matched cloning control
of the same note (BC0 continued on R1's mixture with the 437 collected rows
replaced by fresh expert-distribution rows: 183 successes, 8 violation
episodes, 7 false-commit episodes) shows the direction a pass weighted
toward expert-distribution rows takes; R2's pass moved part of the way in
that direction.

**Rerunning round 2 of this cell under the 2026-10-29 recipe.** The
override is unchanged (the recipe settings default to `cumulative` and
`bc0`); redeploy this commit into the cell so `pipeline.env` and the stage
scripts are refreshed, set the recorded pool-recipe round 2 aside (its
receipts are write-once), and resubmit from the round-2 collection. From
the Windows checkout, with the SSH master up:

```bash
PIPE=/scratch/yx3882/research_full_pipeline_20261022_conventional \
SOURCE_CELL=/scratch/yx3882/research_full_pipeline_20261021_compare \
BASE=$(wsl -- ssh torch "git -C /scratch/yx3882/research_full_pipeline_20261022_conventional/source rev-parse HEAD") \
BRANCH=claude/eager-fermat-0ca9ms OVERRIDES=conventional_only_20261022.env SUBMIT=0 \
bash research/hpc/full_pipeline_20260907/deploy_cell_from_windows.sh $(git rev-parse claude/eager-fermat-0ca9ms)
```

```bash
wsl -- ssh torch "cd /scratch/yx3882/research_full_pipeline_20261022_conventional && mv out/r2 out/r2_pool_20261023 && FROM=r2c CHAIN='r2c r2t r2e' ALLOW_PARALLEL_CHAIN=1 bash submit_pipeline.sh"
```

Round 2 then collects with R1 as before (so `out/r2/collection/` is the
same on-policy draw, re-collected), mixes D2 with all 437 round-1 rows
against a D0 sample of the same size (about 1,738 rows, 435 steps), trains
from the BC0 adapter, and the paired evaluation again scores R2 against R1
on the 192 test roots. `out/r2/collection/mixture_report.json` records
`prior_onpolicy_selected` (437), `prior_onpolicy_retained_fraction` (1.0)
and `label_conflicts`. For the follow-up tables, set
`out/extra/evaluations/hard_r2_control` and `validation_r2_control` aside
and resubmit them (`ONLY="hard_r2_control validation_r2_control" bash
submit_extras_20261023.sh`); the control and the BC0/R1 evaluations are
unchanged, and `followup_tables.py` then re-applies the round selection.

## 5. What the recipe does not change, and what to do about it

- **Round selection.** The pipeline still reports the last round's
  adapter. DAgger returns the best policy over rounds on held-out data;
  the 160 development roots are what the cells report, so selecting on
  them would be selection on the test. A small root-disjoint validation
  draw (for instance 30 roots, like the release path's development
  holdout) evaluated closed-loop after each round would make the choice
  honest; until then, read rounds by their paired per-root record and keep
  the frozen suite for the claim.
- **Quarantined learner-state labels.** The rank-one proof and the offline
  audit drop labels at ambiguous states. The ambiguity design already
  admits a ladder-first non-correction proposal; extending that to the
  remaining quarantine classes (or carrying the deferred expert actions as
  a set-valued label) would give the student a correction where it fails.
- **Evaluation noise.** Report gained and lost roots with their failure
  modes, as the cell records do, and treat a one-root total difference as
  within the decoding noise measured across nodes.
