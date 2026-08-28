# Occupancy exposure-curve cell (2026-08-28)

This is a research-only DAgger-1 experiment. It does not run DAgger-2 and it
does not alter or reinterpret the frozen 2026-08-27 screening cell.

The cell trains two model lanes across three occupancy definitions from the
same from-base initialization recipe and seed `20260823`. Each arm uses its
model lane's current audited D1 universe; the common 66-root held-out assignment
is frozen before the inclusion rule is applied. In the pinned inputs, the root
assignment is also identical across model lanes (canonical split digest
`62cbc48f45dadcea6b3635d072ecb13da0e4c60c0fe143ba194818967dae888b`).

| Arm | Model | Inclusion | Train rows | Validation rows | Two-pass updates | Milestone steps (0.75/1.0/1.5/2.0) |
| --- | --- | --- | ---: | ---: | ---: | --- |
| A | E2B | selective | 1,358 | 340 | 680 | 255 / 340 / 510 / 680 |
| B | E2B | learner-full | 2,566 | 656 | 1,284 | 482 / 642 / 963 / 1,284 |
| C | E2B | full occupancy | 3,630 | 915 | 1,816 | 681 / 908 / 1,362 / 1,816 |
| D | 12B | selective | 1,313 | 350 | 658 | 247 / 329 / 494 / 658 |
| E | 12B | learner-full | 2,533 | 649 | 1,268 | 475 / 634 / 951 / 1,268 |
| F | 12B | full occupancy | 3,576 | 924 | 1,788 | 671 / 894 / 1,341 / 1,788 |

Milestones are the first optimizer boundary at or above the requested pass.
The receipt records exact sampled rows and realized passes, so the small
rounding overshoot at non-divisible 0.75/1.5 boundaries is never hidden. One
and two passes end exactly at corpus boundaries.

## Occupancy and teacher semantics

The aggregate builder never uses learner-action validity as a filter. Invalid
learner-action states therefore remain whenever they belong to the requested
occupancy arm: the clean inputs retain 5/11/15 such rows for E2B A/B/C and
8/13/22 for 12B D/E/F. This is not an instruction to widen the selective arm
beyond learner recovery states.

Audit-failed teacher targets are excluded and retained in immutable abstention
files (49 E2B, 18 12B). Rows with no preferred teacher target are separately
reported (1 E2B, 9 12B); they cannot enter any supervised arm. Every aggregate,
abstention file, prepared-token exposure receipt, and external restart contract
is content-bound by the CPU build.

## Training and restart semantics

Every arm starts from its pinned base model, uses effective batch size four,
and trains until two complete sampled-row passes. Its linear learning-rate
schedule is indexed by cumulative sampled rows divided by `2 * N_train`, not by
a nominal shared optimizer budget. Each milestone stores an adapter-only
checkpoint plus exact cumulative rows, input tokens, supervised target tokens,
and optimizer updates.

Trainer restart checkpoints are requested every 25 updates and at every
milestone; only two periodic restart trees are retained per attempt. A restart
tree is eligible only after its exposure ledger and an atomic, file-hashed
completion manifest agree with Trainer state and the arm's immutable external
contract. A Slurm requeue scans earlier attempts for the highest valid tree and
restores optimizer, scheduler, RNG, dataloader position, and exposure counters.
Attempt-level allocated-GPU receipts remain in the final training gate because
a requeue may move an arm between GPU families.

After training, the same GPU job evaluates all four unguarded milestones on the
same 65-scenario suite. One dependent CPU job per arm replays and physically
audits all four evaluations with an eight-hour allocation. A completed arm therefore represents four
physically audited curve points, not merely four saved adapters.

## Scheduler union

The user-authorized routing is hard-required for both lanes:

```text
a100|h100|h200|rtx6000
```

No script selects a partition. The exact applied Slurm constraint and observed
device identity are attested for every attempt. This deliberately overrides the
handoff document's H200-only proposal: the experiment remains split-, exposure-,
recipe-, and evaluation-matched, but it is not a same-hardware factorial unless
the resulting receipts happen to show matching devices. Hardware differences
must remain an explicit numerical confound.

## Fail-closed launch

Deploy one committed source archive, copy `config.example.json` to an immutable
path, replace every placeholder, and compute the source and offline-model tree
digests with `build.py`. Then submit all six arms through WSL OpenSSH:

```bash
export CELL_CONFIG=/absolute/path/cell.config.json
export CELL_PYTHON=/scratch/yx3882/envs/dagger12b_overlay/bin/python
export CELL_SELECTED_ARMS=A,B,C,D,E,F
bash research/hpc/exposure_curve_20260828/submit.sh
```

`CELL_SELECTED_ARMS` is mandatory so a failed arm can be retried in a fresh
cell root without silently duplicating other runs. A normal experiment uses all
six. The submission receipt must contain exactly one CPU build, one GPU arm,
and one dependent CPU audit for every selected arm. `submit.sh` requests the
four-family union directly and rejects any narrower scheduler receipt.
