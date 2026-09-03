# Numerical-foundation repair of the agentic estimator tools (2026-09-03)

This note records the defects found in an external review of the MCP / deployment
provider toolchain, what was verified empirically before changing code, what
was fixed, and what the fix means for previously collected DAgger evidence.
All checks below run on IEEE case14 through the same code paths the deployment
providers use (`tools/test_numerical_foundations.py`).

## Verified defects

### 1. Global chi-square test used the wrong statistic (confirmed, severe)

`mcp_server/matpower_server._wls_json` reported `global_residual_sum = sum(r_i^2)`
where `r_i = |e_i| / sqrt(Omega_ii)` are the *normalized* residuals returned by
`tools/lagrangian_port`.  The provider compared that value with
`chi2(m - n)` at `chi2_alpha` to decide `no_material_anomaly_remaining`,
`globally_resolved`, `remaining_anomaly_score`, and candidate `global_progress`.

Normalized residuals are correlated and their squared sum has expectation
about `m`, not `m - n`.  Monte Carlo on case14 with the corpus noise model
(`sigma = 0.001` Vm, `0.01` injections and flows; `m = 122`, `n = 27`,
`dof = 95`), 400 clean draws:

| statistic | mean | false alarms at alpha = 0.05 | at alpha = 0.01 |
| --- | --- | --- | --- |
| `sum(r_norm^2)` (old) | 122.5 | 57.0 % | 31.5 % |
| `e' R^-1 e` (new) | 95.4 | 4.3 % | 0.5 % |

Fix: `lagrangian_port.lagrangian_m_singlephase_details` returns the raw residual
and `wls_objective = e' R^-1 e`; `_wls_json` now sets `global_residual_sum` to
that objective and exposes `sum_normalized_residual_sq` for inspection only.
`trace_protocol.build_global_metrics` refuses to fall back to
`sum(r_norm^2)`.  The measurement-correction wrapper also reports the
`wls_objective` of its final solve.

### 2. NLM branch-parameter Jacobian had wrong signs (confirmed)

`tools/lagrangian_port` used `dP_i/db = +Vi Vj sin(d)` and
`dQ_i/dg = +Vi Vj sin(d)` (and the mirrored signs on the `j` end).  Against a
finite difference of the actual measurement model the closed form was off by
150 to 190 percent of the derivative scale on every plain case14 line, while
the separate copy in the parameter corrector matched to 1e-8.  The MATLAB
reference `mcp_server/LagrangianM_singlephase.m` carries the same error
behind "CORRECTED LINE" comments (and is not runnable as checked in); it is
now marked archive-only.

### 3. NLM ignored branch status, tap ratio, and phase shift (confirmed)

The derivative loop treated every row as an energized, unit-tap, zero-shift
line.  Out-of-service branches received a nonzero sensitivity and case14's
three off-nominal-tap transformers were differentiated against the wrong
model.  `build_lambda_evidence` also labelled the two per-branch multipliers
as `"from"` / `"to"` terminals; they are the `R` and `X` multipliers of the
same line.

### 4. Latent crash in the corrector's finite-difference fallback (found here)

`calculate_param_jacobian_for_line_fd` clamped the lower stencil sample to a
positivity floor, so a zero-resistance transformer produced a non-positive
stencil width and raised.  Any parameter correction on case14 rows 8-10 would
have failed.

### 5. Smaller items (confirmed)

* `scripts/quick_check.py` passed `suspect_group=[k + 1]` although the tool
  boundary already converts 0-based indices, so it corrected the neighbour of
  the detected meter.
* `correct_topology` accepted any integer status; `2` or `-1` would scale or
  reverse the branch admittance.
* `Harmonics` admittance builders ignored `BR_STATUS` and assumed consecutive
  1-based bus numbers; HSE acceptance was a THD threshold with no comparison
  against a no-source model, and the THD denominator came from the case's
  planning voltages rather than the observed fundamental.

## Changes

* New `tools/branch_param_jacobian.py`: one closed-form `d h / d [R_k, X_k]`
  in complex notation that is exact for status, tap, shift, and line charging,
  plus a central finite-difference cross-check with no positivity clamp.
  Both the NLM (`all_branch_rx_jacobian`) and the multi-scan corrector now
  call it; the corrector's old helpers are thin wrappers.
* `tools/lagrangian_port.py`: shared Jacobian; `lagrangian_m_singlephase_details`
  returns raw residual, WLS objective, estimated state, `dof`, iterations;
  degenerate multiplier columns report `0`, not `NaN`.  The MATLAB-compatible
  5-tuple wrapper is unchanged.
* `tools/correct_parameter_group_multi_scan_port.py`: convergence additionally
  requires a relative `[R, X]` update below `tol_param_rel` (default 1e-3), a
  non-increasing weighted objective, and no parameter driven from a nonzero
  model value down to the positivity floor; optional `diagnostics` dict
  reports objective reduction and termination reason.
* `mcp_server/matpower_server.py`: raw-objective payload (see above); HSE
  payload adds `null_model_sse`, `best_model_sse`, `sse_reduction_vs_null`,
  `fundamental_voltage_source`, and accepts observed fundamental Vm.
* `psse_env/providers/matpower.py`: `correct_topology` fails closed unless
  status is exactly 0 or 1; `get_topology_context` additionally screens every
  out-of-service branch as a close hypothesis (`hypothesis_source =
  "out_of_service_status_enumeration"`), since an open branch has zero R/X
  sensitivity and can never surface through the multiplier ranking;
  `run_hse` requires `sse_reduction_vs_null >= hse_min_sse_reduction`
  (default 0.5) in addition to the THD threshold and passes the observed
  fundamental Vm.
* `trace_protocol.py`, `schema/sft_trace_decision_schema.json`,
  `psse_env/dagger/sft_audit.py`: `top_lagrange[*].terminal` becomes
  `top_lagrange[*].parameter` with values `R`, `X`, `unknown`.
* `Harmonics/ieee14_verification.py`, `Harmonics/hse_utils.py`: honour
  `BR_STATUS`, resolve bus numbers through the bus-number column.
* `scripts/quick_check.py`: pass 0-based suspect index.

## Ground-truth test suite (`tools/test_numerical_foundations.py`)

26 tests, all passing:

* analytic vs finite-difference Jacobian on every case14 and case9 branch,
  a phase shifter with line charging, an out-of-service branch, and the
  review's spot-check operating point;
* NLM top-1 localization of injected reactance errors, zero multiplier on an
  open branch;
* clean-noise false-alarm rate of the raw objective (100 draws) with the old
  statistic shown to exceed the threshold in more than 30 percent of draws;
* grouped measurement correction recovering one, two, and three simultaneous
  gross errors (including a voltage channel) to within 4 sigma while leaving
  healthy meters untouched and clearing the global test;
* multi-scan recovery of coupled R/X, X-only, and zero-resistance transformer
  reactance errors from observable-only starts to within 5 percent;
* topology identification and repair in both directions (modeled closed /
  truly open via the multiplier ranking; modeled open / truly closed via the
  new status enumeration), with the derived case differing only at the
  target row, and rejection of non-binary status;
* HSE: open branch removed from the harmonic Ybus, three single-source cases
  localized with more than 90 percent null-model SSE reduction, a no-source
  case failing the null test, localization under an open-line topology, and
  observed-Vm THD scaling.

The scenario-generator test `test_known_ambiguous_parameter_root_is_rejected_by_dominance_gate`
encoded the old ranking `[2, 1]` for corpus row `pe_428232230768`, whose true
fault is on line 1.  With the corrected derivatives the true line ranks first
(`[1, 2]`); the root is still rejected as non-dominant.  The expectation was
updated.

## Implications for existing DAgger evidence

* **No-error canaries and near-threshold "globally resolved" labels** were
  produced under a statistic that rejects clean data more than half the time
  at a nominal 5 percent.  Those rows need regeneration under the corrected
  gate before they are used as expert targets.
* **Parameter and topology teacher decisions** consumed branch multipliers
  with wrong signs and without status/tap handling.  Line inventories,
  dominance ratios, and measurement-versus-branch routing derived from them
  should be regenerated or re-audited.
* **Measurement-correction executor outputs** were numerically sound; only
  their global acceptance labels are affected.
* **HIF exact-position claims** remain excluded, consistent with the existing
  observability study.

The immediate next step is a deterministic expert-validation matrix under the
corrected tools, not another DAgger round.

## Follow-up: frozen parameter routes under the corrected multipliers (2026-09-03)

`test_every_frozen_parameter_route_repairs_then_hands_off` (30 frozen
parameter and measurement+parameter roots) passed before this repair and
failed on two roots after it, both by committing a parameter correction on a
healthy line.  Replaying them against the pre-repair commit isolated two
different mechanisms.

**`r0_8c0755fce51c` (forced_error_recovery): a correct repair rejected by a
marginal target test.**  After the meter fix the true line (row 6) is the
top multiplier at 11.8.  Correcting it brings the WLS objective from 208 to
79 against a 130 threshold and leaves no other meter or branch suspect, but
the corrected line's own multiplier settles at 3.13 against the 3.0
per-branch cutoff, so `target_fixed` is false and the candidate was
rejected.  The expert then walked down the ranking: healthy row 3 was
accepted as partial (its own multiplier is small by construction once it is
adjusted, and the objective moved 35 percent) and healthy row 2 as final.
Before the repair the same correction reported a multiplier of 2.10 and
passed; the corrected Jacobian changed the scale, not the physics.  Fix: a
parameter candidate whose own solve passes the global test, leaves no other
suspect, and whose only crossing is its target multiplier inside the
existing 1.25 tolerance band is accepted as final
(`CandidateQualityOracle`, progress class
`observable_resolved_marginal_target`), with the same rule mirrored in the
observable teacher's disposition reconstruction so the teacher and the
environment cannot deadlock on it.  The chi-square test is the
goodness-of-fit statistic; one normalized multiplier among all branches
exceeding a 3-sigma cutoff by a few percent on a clean solve is not
evidence against the repair.

**`r0_b8173b30f6a6` (invalid_action_recovery): adjacent-line ambiguity.**
The true fault is on Line.3-4 (row 5); with the meter error still present
the healthy neighbour Line.2-3 ranks first at 21.6 against 19.0, a
dominance ratio of 1.14.  Correcting either line explains the data equally:
338.5 versus 340.7 with the meter error, 86.9 versus 90.1 (both resolved)
after it.  No observable acceptance rule separates them, and the production
factory sets `BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD = 1.0`, which
disables the dominance gate, so the expert commits the top line.  Before
the repair the wrong-sign multipliers happened to rank the true line first.
This root is unidentifiable under the corrected physics and must be
re-audited in the re-freeze this note already calls for; the safe expert
behaviour for a non-dominant ranking is to escalate rather than commit a
healthy-line adjustment, which is a release-policy decision.

## Not changed (out of scope for research use)

Legacy MATLAB scripts (`Transmission/SE.m`, `SEwithtopo.m`, `WLAV.m`) and the
legacy three-phase WLS remain as archived references; the deployed estimators
are the Python ports.  Multi-source harmonic estimation is still not modelled.
