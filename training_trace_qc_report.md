# Training trace QC report
Repo: `/mnt/data/repo2/psse_agent-feature-balanced-sft-dataset`
## Key findings
- Balanced trace train split has `2003` samples and class counts `{'no_error': 401, 'measurement_error': 402, 'parameter_error': 407, 'topology_error': 397, 'harmonic_anomaly': 396}`.
- Default old train split has `1400` samples and class counts `{'parameter_error': 350, 'no_error': 350, 'measurement_error': 350, 'topology_error': 350}`.
- `three_phase_imbalance` is absent from the balanced train split class counts.
- In balanced traces, `global_residual_threshold` is null for `2003/2003` final targets and `global_residual_ratio` is null for `2003/2003`.
- Importing `scripts/trigger_hse.py` from the repo fails here with `No module named 'matlab'`, so the builder fallback leaves `_chi2_threshold=None`.
- Using the builder's own dof approximation (95 for IEEE-14 full vector), chi-square 95% threshold is `118.752`; `228/401` no-error balanced traces have `global_residual_sum` above that threshold.
- Balanced traces use very high numeric precision: average decimal digits in the first 20 user `z_obs` numbers is `15.95`.
- In a local compaction test, replacing the full WLS tool return with `{success, global_residual_sum, global_residual_threshold, top_residuals[5], top_lagrange[5]}` shrinks the average first WLS tool message from `3148.2` chars to `410.7` chars (saving `87.0%`).
- Rounding all decimals in balanced traces to `6` digits reduces average serialized chars per sample from `23720.6` to `13922.9` (saving `41.3%`).
- Rounding all decimals in balanced traces to `4` digits reduces average serialized chars per sample from `23720.6` to `12084.0` (saving `49.1%`).
- Rounding all decimals in balanced traces to `3` digits reduces average serialized chars per sample from `23720.6` to `11054.7` (saving `53.4%`).
- Rounding all decimals in balanced traces to `2` digits reduces average serialized chars per sample from `23720.6` to `10009.1` (saving `57.8%`).

## Dataset summaries
### Balanced train
- avg messages/sample: `7.40`; median `7`
- avg total chars/sample: `24937.0`
- avg user chars: `2880.2`; avg tool chars: `2864.1`; avg final chars: `1641.0`
- tool call histogram: `{1: 401, 3: 799, 2: 803}`
- tool names: `{'wls_from_path': 2802, 'correct_measurements_from_path': 402, 'correct_parameters_from_path': 407, 'correct_topology_from_path': 397, 'run_hse_from_path': 396}`
- error_family value/type counts: `{'no_error': 401, 'measurement_error': 402, 'parameter_error': 407, 'topology_error': 397, 'harmonic_anomaly': 396}` / `{'str': 2003}`
- global_residual_sum median/p90/min/max: `563.639` / `61621.968` / `75.890` / `1174363.120`
- top_residual list lengths: `{5: 2003}`; top_lagrange list lengths: `{5: 2003}`
- z length histogram: `{122: 2003}`

### Default old train
- avg messages/sample: `7.45`; median `7`
- avg total chars/sample: `10792.1`
- avg user chars: `1239.2`; avg tool chars: `1464.6`; avg final chars: `151.1`
- tool call histogram: `{2: 383, 1: 350, 3: 667}`
- tool names: `{'wls_from_path': 2067, 'correct_parameters_from_path': 350, 'correct_measurements_from_path': 350, 'correct_topology_from_path': 350}`
- error_family value/type counts: `{'parameter_error': 350, 'no_error': 350, 'measurement_error': 350, 'topology_error': 350}` / `{'str': 1400}`
- top_residual list lengths: `{}`; top_lagrange list lengths: `{}`
- z length histogram: `{122: 1400}`

## Per-class char burden in balanced train
- `no_error`: median tool calls `1`, median tool chars `3194.0`, median final chars `1395.0`
- `measurement_error`: median tool calls `3`, median tool chars `12124.0`, median final chars `1678.5`
- `parameter_error`: median tool calls `2`, median tool chars `3330.0`, median final chars `1525.0`
- `topology_error`: median tool calls `3`, median tool chars `8580.0`, median final chars `1700.0`
- `harmonic_anomaly`: median tool calls `2`, median tool chars `4248.0`, median final chars `1912.0`

## Notable schema differences
- Balanced traces final target uses nested `{"verdict", "evidence", "suspect_location", "action", "summary"}`.
- Default old traces final target uses flat fields like `has_error`, `error_family`, `suspect_location`, `recommended_tool`, `confidence`.
- Old training code contains a warning helper about `decision_basis` being requested in the system prompt but omitted in final JSON targets.
