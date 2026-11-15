# The minimal environment: what was built and the first numbers (2026-11-15)

Branch `claude/rl-project-restructure-6526b4`, on top of `claude/open-rules-20261112` (ce6c0e6). Code:
`psse_env/minimal/` and `research/minimal_env/`. The contract of the environment is in `psse_env/minimal/README.md`;
this note records why it exists, how each point of the review was handled, and what the first runs show. All runs
here are local CPU runs. Nothing was trained and nothing was sent to the cluster.

## 1. Why

The review of the open-rules work (2026-11-15) read the clone's result as two facts at once. On the 300 development
cases the adapter cloned on truth-guided rows solved 163, against 186 for S0 in the same rule-relaxed environment:
it gained on the 63 cases only truth-built sequences solve (6 to 20) and lost more on the 184 the rule expert solves
(180 to 143). It also made 54 meter corrections after the estimator already passed, all on healthy meters, all
accepted by the old verifier.

The review's conclusion: the trouble is not which trajectories are imitated but that rules still decide what the
model may try, what is kept and when the episode ends. Its recommendation: keep the numerical tools, hand the
diagnostic decisions to the model, and move the truth to data generation and scoring. Build that as one clean
pipeline beside the old one, not as more switches on it.

## 2. What each recommendation became

| Review | In the code | State |
| --- | --- | --- |
| Separate computation, decision and scoring | `kernel.py` and `versions.py`; `env.py`, `actions.py`, `observation.py`; `scoring.py`, `truth.py` | built |
| `route_status`, `supported_corrections` no longer permit actions | the kernel never calls the provider's `get_*_context`; a target only has to exist | built |
| The verifier returns numbers, not `ACCEPT` / `REJECT` | a correction returns the changes and the estimate before and after; the candidate oracle is not called | built |
| No next-step hints from the controller | no `valid_next_actions`; the process oracle is not instantiated | built |
| A direct finish, no hand-off string | `finish` with `resolved` or `unresolved`, legal at any step | built |
| Format, object, version and numerical checks stay | `actions.validate`, the environment's existence checks, tool failures | built |
| Six actions; a correction saves, executes, re-estimates and reports | `inspect`, `correct_topology`, `correct_parameters`, `correct_measurements`, `undo`, `finish` | built |
| A state that does not converge is returned, not refused | saved and reported as not converged; `undo` works from it | built, tested with a stubbed solve |
| Record model calls, WLS runs, fits and wall-clock | `CostMeter`, `env.cost()` | built |
| A short prompt that lists no rule | `observation.system_prompt`: the task, then what the columns mean | built |
| Current state, actual modifications, action history | the three parts of the user turn | built |
| The same observation constructor in training and at run time | rows and the model policy both call `observation.messages` and `observation.tools` | built, tested |
| No object beyond reach because a rule did not rank it | every bus, line, meter and breaker is in the tables; any breaker can be re-estimated by name | built |
| Re-execute the truth in the new environment | `truth.build_success`, run on the 300 development and 2,550 training cases | built, run |
| Direct sequences for simple cases, search only for coupled ones | breakers, lines, meters first; other orders only when that fails | built |
| States that need no further correction | every path ends with `finish`; a recovery path can stage its error after the last true correction | built |
| Recovery paths after a wrong action | `truth.build_recovery`: wrong correction, `undo`, the rest | built, with simulated errors |
| Keep the cases the expert solves | the selection takes every case | built |
| Do not hand the student the hidden fault list | the environment never receives the audit part; tested | built |
| Supervise the whole decision, no keep/undo emphasis | one row per step: inspect, correct, undo, finish | built |
| Reward: outcome, cost, damage; no workflow terms | `scoring.episode_reward` | built |
| Damage from what changed, not from which target was named | `scoring.physical_comparison` | built |
| `finish` grants nothing | declared, estimator passing and repaired are three separate fields | built |
| Oracle sequences never mixed into RL groups | `rollout.rollout_group` holds only sampled episodes | built |
| KL reference = the new supervised model | the update's reference is the run's starting adapter; an RL run started from the new adapter has it | by construction, no run yet |
| The expert under the new interface, by mechanical translation | `expert.py`, replayed on the 300 development cases | built, run |
| Train the new supervised model | `research/minimal_env/train_sft.py`, rows and selection | data built; **not trained** |
| RL from that checkpoint | reward, rollouts in the RL record format, model policy | **driver loop not built** |

## 3. Checks

### The same numerical tools

| Check | Result |
| --- | --- |
| The estimate's numbers and its bus, line and meter tables equal the old provider's on the same state | equal on 4 generated cases (unit test) |
| The breaker table equals the old topology context's | equal: 73 breakers, 8 flip re-estimations (unit test) |
| The truth's corrections through the old environment (all rules lifted) and through the new one | the same final measurements, case and breaker statuses (unit test) |
| The rule expert's 300 recorded development episodes, translated and replayed | strict audit agrees with the recorded outcome on 300 of 300 (184 solved) |
| Truth-built sequences, old environment against new, 300 development cases | 247 against 247, the same count in each of the 14 classes, the same failure reasons (53, 21, 1) |

### The truth never reaches the model

- A case whose audit part was replaced by another case's gives the same prompts at every step (unit test).
- An execution part that carries a truth field is refused (unit test).
- Metadata outside the evidence profile is dropped before any tool runs (unit test).
- No observation holds a private key, and no prompt holds `ACCEPT`, `REJECT`, `supported_corrections`,
  `route_status` or `valid_next_actions` (unit test).

51 tests: `python -m pytest psse_env/minimal research/minimal_env -q` (about 4 minutes).

## 4. Reference numbers on the 300 development cases

These are the comparisons the review asks to keep. Two rows are still empty.

| Policy | Environment | Solved (strict audit) | Solved (object level) | Actions per case |
| --- | --- | --- | --- | --- |
| Rule expert (guarded variant), recorded | old | 184 | | 12.97 |
| The same episodes, translated and replayed | minimal | 184 | 187 | 6.53 |
| Truth-built sequences, recorded | old, five rules lifted | 247 | | 11.12 |
| Truth-built sequences | minimal | 247 | 247 | 4.24 |
| Supervised model on the new rows | minimal | not trained | | |
| RL from that model | minimal | not run | | |

For orientation, in the old environment with the rules lifted: S0 186, the clone 163.

Reading the rows:

- **The interface change alone changes no outcome.** The rule expert's decisions give 184 in either interface, in
  half the actions.
- **The two audits differ on 3 replayed expert episodes.** In all three the final state is within tolerance of the
  clean one, and the strict audit fails the episode because a correction in effect named a healthy line (2) or
  because an out-of-service line ended attached to another bus section (1). This is the difference the review
  describes between judging the target named and judging what changed.
- **247 is the coverage of these truth-built sequences, not a ceiling.** The other 53 end outside tolerance with the
  true corrections applied (53 line estimates outside tolerance, 21 of them with a line fit that failed).
- **The estimator passing is not the answer.** At the end of the 300 truth-built sequences the estimator passes on
  274. It passes on 40 of the 53 cases that are not repaired, and still alarms on 13 of the 247 that are. The
  replayed rule expert ends 245 episodes on a passing estimate, 69 of them not repaired.

## 5. Cost

Per development case, mean:

| | Actions | WLS solves | Substation solves | Line fits | Meter repairs |
| --- | --- | --- | --- | --- | --- |
| Truth-built, minimal | 4.24 | 2.96 | 10.05 | 1.42 | 0.33 |
| Rule expert replayed, minimal | 6.53 | 4.19 | 6.51 | 3.22 | 0.40 |

The old interface's 12.97 and 11.12 actions are not comparable with the 6.53 and 4.24: one correction there is four
actions. The solves are the comparable quantity. The old records count only the solves the policy asked for (1,040
`run_wls` calls over the 300 expert episodes), not the ones inside its contexts and comparisons, so an old-against-new
table of solves cannot be made from them; the replay's own count is 1,257 WLS solves and 966 line fits.

The truth-built sequences spend most of their substation solves on the inspection at the start of every case (one
estimate and 8 flip re-estimations). That inspection is a choice, see section 8.

## 6. Prompt size

Every digit is a token, so the tables are expensive: one version's bus, line and meter tables are about 4,000
tokens, a breaker table about 2,800, the system prompt 504, the tool list 729.

| Tables shown | Median | 95th percentile | Maximum |
| --- | --- | --- | --- |
| The two most recent versions (default) | 9,282 | 12,049 | 14,276 |
| Every version | 11,451 | 22,403 | 26,596 |

169 rows of 28 development cases (two per class), system and user turns, counted with the local
`unsloth/Gemma-4-26B-A4B-it` tokenizer file: an estimate for the 12B model, not its row gate. For comparison, the
complete-evidence selection of the old environment was estimated at a median of 9,095 and a maximum of 20,662.

## 7. Training data

Built on the 2,550 training cases (`output/compound_rl_20261106/train_ieee14/suite.json`, the suite of the earlier
cloning passes), 21 minutes on 24 local workers.

**Success paths.**

| | Old environment, five rules lifted | Minimal environment |
| --- | --- | --- |
| Cases repaired (strict audit) | 2,091 | 2,091, the same count in each of the 14 classes |
| Orders tried: 1 / 2 / 3 / 6 | 2,052 / 339 / 13 / 146 | 2,052 / 339 / 13 / 146 |
| Actions per case | 11.1 | 4.26 |

The object-level audit calls 2,093 repaired (one more in `meter_rx2` and one in `rx2_distant`). At the end of the
2,550 sequences the estimator passes on 2,272: on 305 of the 459 that are not repaired, and not on 126 of the
repaired ones.

**Recovery paths.** One per repaired case where an error could be staged: 1,855, every one repaired again after the
`undo`. The staged error is a meter correction in 722, a breaker in 681, a line fit in 452. It comes before the first
true correction in 602, after one in 680, after two in 472, after three in 101. In many of them the wrong correction
lowers the chi-square, which is what makes them worth showing: for example a healthy voltage meter moved by 39
standard deviations takes the statistic from 2,861 to 2,008, and the demonstration undoes it.

**Rows.** 16,077 rows, each checked against the schema half of the trainer's row gate when written.

**Selection** (`select_rows`, defaults: every case, a 6% share of each class held out whole):

| | Cases | Rows |
| --- | --- | --- |
| Training view | 2,398 | 14,823 |
| Validation view | 152 | 256 (sampled from 944, balanced over step kinds) |
| Left out | | 305 `finish unresolved` rows on a passing estimate |

Training rows by action: `correct_parameters` 3,219, `correct_topology` 1,989, `correct_measurements` 1,213,
`inspect` 2,814, `undo` 1,738, `finish` 3,850. By kind: success path 9,524 (corrections 4,722, inspections 2,690,
finishes 2,112), recovery path 5,299 (undo 1,738, corrections 1,699, finishes 1,738, inspections 124).

Of the 2,550 cases, 2,091 give all their rows and 459 are beyond the correction tools: those give their true-target
corrections and, in 152 of them, a `finish unresolved` on a final estimate that still alarms.

**Prompt length of the training view** (a 5% sample, 780 rows, the same tokenizer file): median 9,210 tokens, 95th
percentile 11,860, maximum 14,412, about 134 million prompt tokens in all. The complete-evidence selection was
estimated at 96.6 million for 11,061 rows, so a pass over this view is about 1.4 times that one's token volume. This
is my estimate from a sibling tokenizer; the row gate on the cluster gives the real figure.

Checksums: training view `582349e2...`, validation view `e55689a6...` (`selection_report.json`).

## 8. Choices made here that are yours to change

1. **Tables of two versions in the prompt.** Older versions keep their summary numbers and are one `inspect` away.
   Showing every version is one argument (`table_versions=None`) and costs up to 26,600 tokens instead of 14,300.
2. **An inspection of the breakers at the start of every demonstration.** Without it, whether to look at the breakers
   would be decided by the truth (inspect only when a breaker is wrong), which a student cannot imitate from the
   evidence. With it, every demonstration starts with the same action and the breaker table is in view for the first
   decision. It costs one action and 9 substation solves per case, and about 2,800 prompt tokens while that version
   is shown. `--no-opening-inspection` builds the other kind.
3. **Simulated errors in the recovery paths.** The wrong correction is the healthy meter with the largest residual,
   the healthy line with the largest multiplier or the healthy breaker with the largest score. This is my stand-in
   for a student's mistake until a trained model's own failed episodes exist.
4. **`finish unresolved` rows.** Kept only where the final estimate still alarms; dropped where it passes, because
   the prompt cannot tell that state from a repaired one.
5. **The reward weights** in `scoring.RewardWeights`: success 1.0, repaired but not declared 0.5, false `resolved`
   0.5, damage 0.25 per object up to 0.5, 0.01 per action.
6. **The claim of the replayed expert.** The old hand-off closes every episode with the same request, so the replay
   declares `resolved` when the final estimate passes. The expert's number above is `solved`, which does not depend
   on this.
7. **Tool limits left as they are.** A breaker flip that would isolate equipment (not a line terminal, a split or a
   merge) is refused by the breaker tool as before; the line fit and the meter repair are the old ones. Nothing in
   this work changes what the tools can do, so the 53 unrepaired cases stay unrepaired.

## 9. Not done

- **No model was trained.** The rows, the selection and the trainer entry point exist; the trainer entry point has
  not run, and the tokenizer row gate (cluster) has not run on these rows.
- **No RL run, and no RL driver for this environment.** `psse_env/rl/train.py` drives the old environment. What the
  driver needs from this side exists and is tested without a model: grouped rollouts in the RL record format, the
  reward, the model policy.
- **No cluster cell** for either.
- **IEEE 118** is not handled: no breaker model there (the line-status form of `correct_topology` exists and is
  untested on that system), and a version's meter table is about 1,100 rows.
- **The waveform families** (HIF, unbalance, harmonics) are outside this environment.
- The full repository suite was not rerun: no existing module was changed, only two README files.

## 10. Where things are

| What | Where |
| --- | --- |
| Environment and tests | `psse_env/minimal/` |
| Builders, baseline, selection, trainer entry point | `research/minimal_env/` |
| Truth reference, 300 development cases | `output/minimal_env_20261115/truth_reference_dev300/` |
| Rule expert replay, 300 development cases | `output/minimal_env_20261115/expert_dev300/` |
| Demonstrations and rows, 2,550 training cases | `output/minimal_env_20261115/demonstrations_train_ieee14/` |
| Training and validation views | `output/minimal_env_20261115/selection_ieee14/` |
| The four runs' summaries, committed | `docs/minimal_env_20261115/` |

`output/` is ignored by git and sits in this session's worktree
(`.claude/worktrees/dreamy-bassi-27849a/output/minimal_env_20261115/`, 555 MB). The runs were made before the
commit, so their manifests name the parent commit (ce6c0e6); the code that made them is this commit's. The development suite and the rule expert's
recorded episodes were read from the worktree `nostalgic-kalam-be5f7a`
(`output/compound_rl_20261107/ieee14_dev300.json`, `output/compound_rl_20261106/reference_ieee14/teacher/episodes.jsonl`);
the training suite from the main checkout (`output/compound_rl_20261106/train_ieee14/suite.json`).
