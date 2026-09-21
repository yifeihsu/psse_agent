# Shared 40-action episode budget

The active experiment configuration uses 40 actions per episode across expert
data generation, DAgger collection, closed-loop evaluation, and transactional
environments. The Python default is `DEFAULT_EPISODE_ACTION_LIMIT` in
`psse_env/episode_budget.py`.

The full-pipeline and diagnostic-round shell configurations derive their stage
limits from `EPISODE_MAX_STEPS`, which defaults to 40. They rebind the stage
aliases after reading override files so old independent collection/evaluation
settings cannot silently produce different horizons. Low-level runners retain
explicit positive overrides for deliberate experiments and unit tests; the
current production study contract uses 40.

## Counting and enforcement

- Each attempted agent action consumes one step, including an invalid or failed
  action. A successful finalization is an action within the same allowance.
- Evaluator setup/intervention actions consume part of the episode allowance.
  For example, three setup actions leave 37 actions for the evaluated policy.
- The runner sets the environment's limit before reset. The remaining budget
  exposed to the policy and stored in transitions follows that limit.
- The transactional environment permits action 40 and refuses action 41 before
  dispatching a provider call. Reaching the limit does not manufacture a
  successful finalization, physical recovery, or operator escalation.
- Independent counterfactual branches inherit the current budget and then
  account for their own attempted actions.
- Legacy agent evaluators use the same default and dispatch at most one tool
  per assistant action. Batch retries do not restart a partially executed
  episode with a fresh allowance.

New evaluation artifacts declare `action_budget_scope: all_episode_actions`.
Their gate and study metrics use total episode steps, including setup. Older
artifacts retain their original counting convention when audited under their
historical policy.

## Historical experiments and training

The default study template is now
`psse_env/dagger/studies/dagger_multiseed_study_v2.json`, paired with the current
40-action policy `bc0_closed_loop_hard_gate_v4`. The original v1 study and its
24-action policy are preserved for explicit historical validation; see
`psse_env/dagger/studies/README.md`.

Optimizer update counts, epochs, token limits, and historical result files are
unchanged. These edits affect newly started runs using this checkout. They do
not alter a running process, regenerate earlier training data, or reevaluate
existing checkpoints.

## Validation

Regression coverage checks the 40th/41st-action boundary, failed actions,
finalization, reset and clone behavior, shared launcher defaults, environment
binding, remaining-budget observations, intervention accounting, historical
study preservation, and total-action evaluation gates. The validation receipt
is `output/episode_action_budget_20260916/validation_summary.json`.
