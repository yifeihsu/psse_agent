# The second supervised pass of the minimal environment (prepared 2026-11-21)

The first pass's cell (`research/hpc/minimal_sft_20261115/`) with other rows. Asked by the user on 2026-11-21
("Yes, prepare it"), after the stopping-state diagnostic (`research/hpc/minimal_diag_20261120/README.md`) showed
where the first supervised model stops:

* after the first of two breaker corrections it does not look at the breakers again (pattern A, 17 of the 49
  unrepaired solvable development cases); the first demonstrations looked there only when the truth knew another
  breaker was wrong (300 of 300 two-breaker cases, none of 1,000 others);
* in cases with a breaker error and a line or meter error it corrects what it sees first and finishes without
  looking at the breakers again (pattern B, 7 of 49), the step on at a probability of about 0.0003, out of reach of
  sampling;
* it answers a refused line fit by finishing (117 of the 144 training cases of A had one);
* it writes `CB_2R4_R5` for `CB_2R4_2R5` (245 of 263 calls).

**State: prepared, nothing deployed.** Deploying, each smoke and each submission wait for the user's word.

## The rows

The second demonstration build (`research/minimal_env/build_demonstrations.py`, options of 2026-11-21; the
numbers are in `docs/minimal_env_20261115.md`, section 14):

* `--looks every_version`: a demonstration looks at the breakers (`inspect`, view `breakers`) before acting on any
  version whose breakers it has not looked at, `finish` included, so looking is never a decision the truth makes;
* `--refusals`: a refusal path per repaired case: the true corrections up to the first breaker correction, then on
  that version, before any look, a fit of the healthy line with the largest multiplier, kept when the tool refuses
  it; the rows begin after the refused call (context, never a target);
* `--breakers-last`: for a repaired case with a breaker error and another, the same corrections with the breakers
  last, the look that shows the breaker, and a refit of what was corrected on the wrong topology; kept when it
  repairs by both audits.

`research/minimal_env/select_rows.py` then takes every case as before; `--repeat-breaker CB_2R4_2R5=3` writes the
39 training rows that correct that breaker three times.

**Built 2026-11-21 at e95db77** (`output/minimal_env_20261121/`, not in git; numbers in section 14 of the doc):
33,181 rows from the 2,550 training cases (first build 16,077): 2,550 success paths, 1,855 recovery paths, 912
refusal paths and 635 breakers-last paths; the look follows 1,600 of the 1,600 accepted breaker corrections of the
success paths (first build 300); 6.11 actions a success path (4.26).  The selection: 30,960 training rows (14,823)
on the same 2,398 training cases, 256 validation rows on the same 152; about 369M prompt tokens (137M), median
12,327 and longest 16,477 by the sibling-ratio estimate; train sha256 aa0e7985..., validation 67ffcb58....

**Cost of a pass** (my estimate): 7,740 optimizer steps at about 47,700 tokens a step (the first pass: 3,706 at
about 37,000, 51 s a step, 49.5 hours): about 66 s a step and 140 hours from the base model.  `submit_cell.sh`
counts 75 s a step: 16 links of 12 hours.  The upload is 689 MB.

## Starts

`SFT_START` (cell.env): `base` (default: a new LoRA on the base model at 1e-4, as the first pass, so the two passes
differ in their rows alone), `sft1` (the first pass's adapter, read in place, at 3e-5: cheaper, starts from its
habits) or `r1`. Each start keeps its output apart (`out/pass_<start>`).

## Order of work, each cluster step on the user's word

1. Build and select the rows on Windows (done here; under `output/minimal_env_20261121/`, not in git).
2. Deploy: `bash research/hpc/minimal_sft_20261121/deploy_from_windows.sh <commit40>`; the default selection folder
   is this worktree's `output/minimal_env_20261121/selection_ieee14`.
3. Smoke: `wsl -- ssh torch bash -s -- smoke [START] < research/hpc/minimal_sft_20261121/submit_cell.sh`.
4. The row gate, the pass and its evaluation: `... -- all [START] < ...submit_cell.sh`.
5. Status: `wsl -- ssh torch bash -s < research/hpc/minimal_sft_20261121/status_cell.sh`.

Everything else (the stages, the row gate's cache, the evaluation on the 300 development cases, the receipt) is the
first pass's; its README has the details and what its smoke showed.

## Files

As the first pass's cell; the differences: the work directory `/scratch/yx3882/minimal_sft_20261121`, the job names
`ms2-*`, the `sft1` start (`SFT1_ADAPTER`, `START_ADAPTER` in the receipt), 75 s a step in the link count (the first
pass ran 51 s a step on shorter rows), and the source cloned from the diagnostic cell's.
