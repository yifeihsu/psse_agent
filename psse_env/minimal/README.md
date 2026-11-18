# The minimal environment

`psse_env.minimal` is a second, independent way to run a diagnosis episode. It was built on 2026-11-15 after a
review of the open-rules work. The old environment (`psse_env.transactional_env`) is untouched and stays the way to
reproduce every earlier result.

The idea in one sentence: **the numerical tools compute, the model decides, and the truth is used only to build
training data and to score a finished episode.**

## Why

In the old environment the rule expert's procedure is part of the environment: a correction must be on a context's
list, a verifier accepts or rejects it, a commit follows the verdict, and the controller suggests the next step. The
`open_rules` option lifted five of those rules, but the prompt then had to explain which rules were lifted, the
candidate lifecycle (context, correction, verification, commit) was still the task, and the truth-built sequences
still ran through it. A model trained there learns the old workflow with exceptions.

Here there is no workflow to learn. Results from the first runs are in `docs/minimal_env_20261115.md`.

## Three layers

| Layer | Module | Does | Never does |
| --- | --- | --- | --- |
| Numerical execution | `kernel.py`, `versions.py` | state estimation, substation estimation, measurement repair, line fit, breaker flip, saving and restoring states, counting cost | rank a target, offer or withhold a correction, accept or reject one, say what to do next |
| Model decision | `actions.py`, `env.py`, `observation.py`, `policy.py` | take one action from the model, check its form, run it, return what happened | see the truth, or a recommended next step |
| Training and evaluation | `scoring.py`, `truth.py`, `rows.py`, `rollout.py`, `expert.py` | build demonstrations from the truth, score a finished episode, compute the reward | approve or refuse an action while the model acts |

The kernel reuses the numerical routines of `psse_env.providers.matpower.MatpowerDeploymentProviders` (the same
solves, repairs and fits) and the complete tables of `psse_env.providers.complete_evidence`. It does not call that
provider's `get_*_context` methods, `run_wls` or the candidate oracle, which is where the top-5 cuts, the dominance
test, `supported_corrections`, `route_status` and the accept/reject verdict live.

## The six actions

| Action | Arguments | What happens |
| --- | --- | --- |
| `inspect` | `view` (`breakers` or `estimate`), optional `version`, optional `flip` | `breakers`: the substation estimate of a version with its table of every breaker; the breakers named in `flip` are re-estimated with their status changed. `estimate`: the tables of a version again. Changes nothing. |
| `correct_topology` | `breaker`, `status` (`open` / `closed`) | sets the reported status of one breaker; the operator model is rebuilt. On a system without a breaker model the arguments are `line` and `status` (`in` / `out`). |
| `correct_parameters` | `line` | re-fits the line's series R and X from the repeated scans |
| `correct_measurements` | `meters` | repairs the named measurements as one group; the tool computes the values |
| `undo` | optional `version` | returns to a saved version (by default the one before the latest correction in effect) |
| `finish` | `outcome` (`resolved` / `unresolved`) | ends the episode on the current version |

A correction is one reversible action. The environment saves the state, runs the tool on the object the model named,
re-estimates the new state, and reports what actually changed and what the estimator now shows. There is no
candidate, no verification step and no commit: a correction stays in effect until the model undoes it. Whether to
keep it is the model's decision, made from the numbers.

**What the environment checks** is the execution contract only: the call parses, the tool exists, the arguments have
the declared types, the named version, line, meter or breaker exists, and the tool produced a usable change. A call
that fails one of these is a tool error; it changes nothing and counts as an action. Error codes:
`unparseable_action`, `unknown_tool`, `invalid_arguments`, `unknown_object`, `no_change`, `tool_failure`.

**What it does not check:** whether a context was read first, whether the target is on some list, whether the
estimator already passes, whether a change lowered the chi-square. A state whose estimate does not converge is saved
and reported as not converged; the model may undo it or go on.

`finish` can be called at any time and grants nothing. The scorer keeps three things apart: what the model declared,
whether the estimator passes, and whether the errors were actually repaired.

## What the model reads

`observation.messages(observation)` and `observation.tools(observation)` build the prompt. Training rows
(`rows.py`) and the run-time policy (`policy.py`) both call these two functions, so the model never reads one layout
in training and another when it acts.

The system prompt states the task and what the columns mean. It names no rule. The user turn has three parts:

1. **History**: every call, its result, and the version it acted on.
2. **Modifications**: for each correction, what changed in the state (read from the saved states, not from the
   tool's report: a breaker status, a line's R, X or service status, a meter reading, or a rebuilt operator model),
   the estimator's numbers before and after, and whether it is still in effect.
3. **Versions**: each saved state's estimate: convergence, chi-square with its threshold, largest normalized
   residual with its threshold; and the tables of every bus, every line (model R and X, their multipliers, flow
   residuals) and every meter (reading, signed normalized residual), plus the breaker table where inspected.

Nothing is ranked, cut at a threshold or summarized. The tables are comma-separated rows.

**One limit, stated plainly.** Every digit is a token, so one version's tables cost about 4,000 tokens and a breaker
table about 2,800. By default the observation shows the tables of the two most recent versions (the current one and
the one before it, which is what a before-and-after reading needs); older versions keep their summary numbers and
their tables are one `inspect` away. `MinimalDiagnosisEnv(table_versions=None)` shows every version.

| Tables shown | Prompt tokens, median | 95th percentile | Maximum |
| --- | --- | --- | --- |
| two most recent versions (default) | 9,282 | 12,049 | 14,276 |
| every version | 11,451 | 22,403 | 26,596 |

(169 rows from 28 development cases, two per class; system and user turns; counted with the local
`unsloth/Gemma-4-26B-A4B-it` tokenizer file, an estimate for the 12B model.)

## Scoring and reward

`scoring.score_env(root, env)` scores a finished episode against the scenario's audit part. It reports:

- `declared`: `resolved`, `unresolved`, or nothing (the budget ran out);
- `estimator_passes`: both tests of the final estimate below their thresholds;
- `physical`: the final state compared with the clean one, object by object, with the tolerances the scenario
  declares. A true error is repaired when its reading, R and X, or status ends within tolerance. A healthy meter or
  line is damaged when it ends outside the same tolerance. `solved` means every error repaired and nothing damaged;
- `strict`: the audit every earlier result was scored by (`release_audit.audit_truth_audited_task_success`), fed
  the corrections in effect at the end. It also fails an episode in which a correction in effect names a healthy
  target, whatever the change did. It is there so a number from this environment can stand beside the old ones.

`scoring.episode_reward` is the first version's reward:

```
reward = outcome - false_resolved - damage - action_cost * actions - tool_error * errors
```

| Term | Default | Meaning |
| --- | --- | --- |
| `success` | 1.0 | solved and declared `resolved` |
| `unclaimed_success` | 0.5 | solved, but declared `unresolved` or the budget ran out |
| `false_resolved` | 0.5 | declared `resolved` on a state that is not solved |
| `damage` | 0.25 each, at most 0.5 | healthy objects left outside tolerance |
| `action_cost` | 0.01 | per action |
| `tool_error` | 0.02 | per refused call, on top of its action cost |
| `tool_second` | 0 | per second of numerical tool time; off until efficiency is the question |

Solving with the whole 16-action budget (0.84) still beats stopping at once (-0.01). There is no term for following
a workflow and no keep-or-undo label: a correction is kept or undone by the model's own later action and credited by
the episode's outcome.

## Cost

A correction here is one action where the old interface needed four (context, correction, solve, commit), so action
counts across the two interfaces are not comparable. `env.cost()` therefore reports, per episode: `actions`,
`wls_solves`, `substation_solves` (the node/breaker estimator, including each flip re-estimation),
`parameter_fits`, `measurement_repairs` and the seconds each kind took.

## Demonstrations

`truth.py` builds training sequences from the truth by running them here. Nothing is taken from an older trace.

- **Success path** (`build_success`): the true corrections, breakers then lines then meters, one action each, then
  `finish`. Other orders are tried only when that one does not end repaired. A case no order repairs ends
  `finish unresolved` and is reported as a limit of the correction tools.
- **Inspection**: the breakers are inspected before a breaker correction on a version where they have not been, and
  once at the start of every case with a breaker model, so that looking at the breakers is not a decision the truth
  made for the student.
- **Recovery path** (`build_recovery`): some true corrections, then one correction on a healthy component, then
  `undo`, the remaining true corrections and `finish`. Only the steps from the `undo` on are training targets. The
  healthy component is the one the shown evidence makes easiest to mistake (largest residual, largest multiplier,
  largest breaker score); this simulates a student's error and can be replaced by a trained model's own failed
  episodes once there are some. When the error is staged after the last true correction, the path is "a wrong
  correction made when everything was already right, undone, then finish".
- **Stopping**: every path ends with `finish` on the state the truth judges.

## Baselines

`expert.py` translates a recorded episode of the rule expert into these actions, step for step, and replays it. The
translation adds no truth and no heuristic, with one exception that is stated in the module: the old hand-off made no
claim about the result, so the replay declares `resolved` when the final estimate passes.

## Commands

```bash
python -m pytest psse_env/minimal research/minimal_env -q
```

```bash
python -m research.minimal_env.build_demonstrations --suite <suite.json> --output-dir <dir> --workers 20
```

```bash
python -m research.minimal_env.select_rows --demonstrations <dir> --output-dir <selection dir>
```

```bash
python -m research.minimal_env.expert_baseline --suite <suite.json> --episodes <rule expert episodes.jsonl> --output-dir <dir>
```

```bash
python -m research.minimal_env.evaluate --suite <suite.json> --adapter <adapter dir> --output-dir <dir>
```

`research/minimal_env/train_sft.py` runs the stock `research-train` on these rows (it keeps each row's own six
tools instead of the old registry), `row_gate.py` runs the trainer's tokenizer gate alone, and `evaluate.py` runs a
trained adapter on a suite. All three need the model or its tokenizer; the cluster cell that calls them is
`research/hpc/minimal_sft_20261115/`, whose README holds its state.

`research/minimal_env/rl_train.py` is the RL driver: sampled episodes of this environment (`rollout.py`), the
outcome reward of `scoring.py`, and the update of the RL series (`psse_env/rl`) on the sampled tokens, with the
supervised model as the KL reference (docs/minimal_env_20261115.md, section 12). It has run with stand-ins for the
model only.

## Settings a model run needs

`policy.MinimalModelPolicy` is the research Gemma policy with this package's renderer, so it reads the same two
environment variables (`psse_env/dagger/preliminary_e2b_eval.py`):

- `RESEARCH_MAX_INPUT_TOKENS` defaults to 8,192, which is below most prompts here. A prompt over the limit is
  refused, not cut, and the episode then ends with nothing declared (`prompt_limit`). Set it to at least 16,384 for
  the default table window, 32,768 with every version shown.
- `RESEARCH_MAX_NEW_TOKENS` defaults to 64. That is enough for every call the demonstrations make. An `inspect` that
  names eight breakers to flip is longer; 128 covers it.

## Not built yet

- The supervised model exists since 2026-11-18: 198 of the 300 development cases, rule expert 184, truth-built
  sequences 247 (`research/hpc/minimal_sft_20261115/README.md`). What follows is still not built.
- An RL run. The driver (`research/minimal_env/rl_train.py`) and its cluster cell
  (`research/hpc/minimal_rl_20261116/`) have met the model in a smoke alone (2026-11-17: one small iteration,
  every check of the update held).
- A paging mode for large systems. On IEEE 118 one version's meter table alone is about 1,100 rows.
- Recovery paths from a model's own failed episodes. (The focused RL run, built 2026-11-18, teaches a failed case's
  truth-built sequence from its first state: `research/hpc/minimal_rl_20261116/README.md`.)
