# Merge-readiness review: compound stage 2 (`claude/nostalgic-kalam-be5f7a`, d6e6acc..6e5c478)

2026-10-28. Reviewed range: base d6e6acc, head 6e5c478. That is 22 commits and 35 files. The stat reads +7,171/-313; ignoring end-of-line changes it is +7,001/-143. The diff and the files were read at the head ref. The reviewed branch was not modified, and nothing under `docs/papers/` was touched. This note is committed on `claude/upbeat-driscoll-ff493f`, which was fast-forwarded to 6e5c478, so its only change is this file. It can be cherry-picked on its own.

Method:
1. A `code-review` skill pass at high effort over the range.
2. A targeted hand pass over the six areas the requesting session named.
3. Runs: the full test suite at the head, the four failing tests at three commits, an expert end-to-end on the conventional 192-root suite at both the base and the head, an observation probe, and a cost probe. Section 3 lists them with counts.

The scratch scripts and logs are outside the repository and are not part of this commit.

## 1. Findings, ranked by severity

Each finding gives its file and line at 6e5c478, the failure scenario, and a suggested fix. Apart from this note, nothing was changed.

### High

**H1. The HPC pipeline refuses the new default teacher.**
- Where: `research/hpc/full_pipeline_20260907/pipeline.env:286`.
- `pipeline_environment()` still accepts only `baseline|ledger|ledger_ranked|ledger_ranked_compare|baseline_compare`. Commit b546d2d changed the default at `pipeline.env:199` to `baseline_compare_topology_clean_flip` and extended the list in `deploy_remote.sh:49`, but left this list alone.
- Verified by running: this file's `pipeline.env` was sourced with `PIPE` pointed at a scratch directory, then `pipeline_environment` was called.
  - With `EXPERT_VARIANT` unset, it returns 2 with `unknown expert variant: baseline_compare_topology_clean_flip`.
  - With the clean-flip variant named, it returns the same.
  - With `EXPERT_VARIANT=baseline`, it returns 0.
- Every sbatch stage calls `pipeline_environment`. A new cell therefore passes `deploy_remote.sh` and then dies in its first stage. The same happens to any cell that names one of the three new variants.
- A classifier-gated cell that names no teacher also stops at `prerequisites.sh:75`, which requires `baseline`. That stop is loud, so it is correct as it stands.
- Fix:
  - Add the three variant names to the case list at `pipeline.env:286`.
  - Add a test that checks the variant lists in `pipeline.env` and `deploy_remote.sh` against `EXPERT_VARIANTS`.
  - Do this together with M1. Until line 286 is fixed, the two override files that name no teacher fail loudly. Once it is fixed, they would switch teacher silently.

**H2. Under the new default, the student's observation reveals the topology family on every existing suite.**
- Where:
  - `psse_env/providers/matpower.py:1996-1997` (the screen is gated only by the provider option).
  - `matpower.py:3708-3712` (`_clean_flip_screen` returns None when the state carries no node/breaker channel).
  - `psse_env/transactional_env.py:862` (the WLS ledger copies `clean_flip` when present).
- `clean_flip` is therefore present in the durable WLS ledger exactly on roots that carry substation telemetry. `prepare_model_policy_observation` keeps that ledger model-visible.
- On the conventional suites, only the topology and measurement+topology roots carry substation telemetry:
  - `development.json`: 56 of 192 roots;
  - torch `validation.json`: 28 of 96;
  - torch `hard.json`: 16 of 193.

  Every other family carries none.
- Verified by running a probe: reset the research environment under scada_only with `PSSE_EXPERT_VARIANT=baseline_compare_topology_clean_flip`, take one `run_wls`, apply the model-visible preparation. The probe covered two roots per family of `development.json`.
  - `clean_flip` was present after the first solve on 4 of 4 topology and measurement+topology roots.
  - It was absent on all 10 others.
  - The policy had requested nothing at that point.
- The project's rule is that availability must never leak the family (the uniform-metadata convention of 2026-10-28). That convention covers future suites. The default, however, applies to any new work, including work on these existing suites. The expert-only checks in the study (validation 96, hard 193) could not see this, because the expert reads `clean_flip` legitimately.
- Fix, keeping the user's choice of teacher:
  - make the clean-flip teacher refuse to run on a suite whose roots do not all carry the node/breaker channel (a check in the suite builder or at environment reset);
  - or, on such suites, keep `clean_flip` out of the model-visible view.
- Reporting a "no channel" status on the other roots would not help: the status itself would name the family.

**H3. `flow_at_noise_lines` enters every student observation under every teacher.**
- Where:
  - `matpower.py:1995` and `:2023` compute and report it on every anomalous non-waveform solve, with no option.
  - `matpower.py:3021` adds `parameter_flow_at_noise_lines` to every parameter context.
  - `transactional_env.py:859` and `:4282` copy both into the durable ledgers.
- Verified by running the same probe under `PSSE_EXPERT_VARIANT=baseline`. Every root's observation after a successful solve now carries a `flow_at_noise_lines` list. It is empty unless the solve is anomalous, and it is non-empty on 3 of the 4 topology roots (lines 17, 12 and 11). The key did not exist at d6e6acc.
- The study's "observable, additive fields; no default route decision changes" (`matpower.py:557`) is true of the expert. It is not true of the student's input.
- Effects:
  - every SFT row and every evaluation prompt built on this code differs from what the BC0/R1/R2 adapters of earlier cells saw;
  - this includes the conventional-only cell behind the frozen PES GM table, if it is ever re-evaluated on merged code;
  - the field is a strong breaker cue (522 of 650 compound breaker roots at the first solve, per the code comment), so it can change student behaviour.
- The cue is computed from SCADA flows and the solve's residuals, not from truth. So it is not a leak; it is a contract change.
- Fix: report it in the ledger only when a run's teacher or verifier option reads it (`topology_first="flow_at_noise"`, `target_first_partial`). Otherwise keep it in the tool metrics only, out of the model-visible ledgers. At minimum, record the observation-contract change in the receipts so that old adapters are never silently evaluated on the new prompts.

### Medium

**M1. Two recorded override envs name no teacher.**
- Where: `research/hpc/full_pipeline_20260907/overrides/ieee57_transfer_20260914.env` and `overrides/ieee118_balanced_20260925.env`. Neither has an `EXPERT_VARIANT` line, and both ran under the former default, `baseline`.
- Today, because of H1, they fail loudly. Once H1 is fixed, they would run `baseline_compare_topology_clean_flip`.
- On case57 and case118 the node/breaker binding is None, so the screen never fires and the teacher becomes plain `baseline_compare`. The bounded comparison then relabels ambiguous parameter rankings: a different D0, with no error.
- The comment at `pipeline.env:193-198` documents the re-run rule ("re-run as recorded with EXPERT_VARIANT=baseline") but does not enforce it.
- Neither file pins `EVIDENCE_PROFILE` either. That predates this branch, but it means the two files no longer reproduce their cells as recorded.
- Fix: add `EXPERT_VARIANT=baseline` to both files, in the same change as H1. Optionally, have `deploy_remote.sh` refuse an overrides file that names no teacher.

**M2. Per-episode cost of the screen.**
- Measured with a probe that attributed every `_node_breaker_estimate` call by call site. Setup: 8 compound roots, one per class (cb2_dangling, cb2_mixed, cb2_split, control_cb, control_meter, control_rx, meter_on_rx, meter_rx2), under the clean-flip teacher in scada_only.
- Results:
  - 61 screens spent 540 node/breaker solves, about 67 per episode. Every one of them came from a policy-visible `run_wls` or `verify_candidate`.
  - The 5 topology contexts were all served from the cache, at 0 solves each.
  - No solve came from the nested `run_wls` calls inside `_screen_topology_correction` or `_assess_measurement_candidate`. The skill pass had suspected those; the measurement refutes it.
- This sample is breaker-heavy, so it sits above the study's suite-wide figure of about 23 per episode.
- The cache works as intended. It is per provider, the evaluator builds a fresh environment for every episode, and FIFO eviction caps it at 64 records. Nothing is shared across forked `ProcessPoolExecutor` workers or GPU evaluation workers, and memory cannot grow across episodes.
- The cost is CPU time inside GPU evaluation stages. Under the torch cluster's low-GPU-utilization cancel thresholds (H100/H200 at 60%, B200 at 70%), stages that already run at 54-67% utilization lose margin.
- Fix (optional): screen only where the teacher routes on it, that is, the active state's solve rather than every candidate verification. Or report the cost in the evaluation receipts so that stage budgets can be set.

### Low

**L1. The `clean_flip` record exposes controller bookkeeping to the student.**
- Where: `matpower.py:3719`, `:3722` and `:3733-3741`.
- The whole dict is copied into the ledger, including `solves`, `cached`, and on failure the raw exception text (`f"{type(exc).__name__}: {exc}"`).
- Effects:
  - In the M2 cost probe, all 61 policy-visible screens reported `cached: false`, so the variation below is rare in practice; what remains is bookkeeping in the prompt.
  - The same physical state renders different prompt text depending on its history: `cached: true, solves: 0` after a topology context on that state, against `cached: false, solves: 9` before it.
  - Exception text that names a content-addressed derived-case path (`<tag>_<sha256>.m`) would put a 64-hex digest into model-visible data, which `find_model_identifier_leaks` rejects at export.
- Fix: copy only `status` (as a fixed vocabulary), `available`, `cb_names`, `screened` and `reported_chi_square` into the ledger. Keep `solves`, `cached` and the error detail in the tool metrics.

**L2. The second structural flip renders from a reference with forced branch statuses.**
- Where: `matpower.py:3246` forces `branch[:, 10] = 1.0` in the binding's reference; `matpower.py:3586` renders a later flip from that reference.
- The first flip, by contrast, re-loads the case file. `operator_model_from_map` documents that "a line the operator already holds out of service stays out".
- So a line held out of service independently of the breaker map would come back in service on a second flip, but not on the first.
- No current root carries such a line: roots are rendered on the planning case. The bug is latent.
- Fix: render from the planning reference with the current case's status column copied back, or document why that column never matters.

**L3. The node/breaker cache key leaves out telemetry and the planning reference.**
- Where: `matpower.py:3678-3685`. The key is the case path (content-addressed), the measurement vector and the reported labels.
- Neither the substation telemetry nor `planning_reference_case` is in it, yet both feed `gse_topology_nlm`.
- No current action changes telemetry without also changing the labels or measurements, so this is latent.
- Fix: add a digest of the synchronized telemetry and the planning-reference path to the key.

**L4. Replays of states recorded before this branch can now fail the topology binding.**
- Where: `matpower.py:3236-3240`.
- A state that was re-rendered by a bus split on pre-branch code has no `planning_reference_case` and a sectioned bus count. Its topology binding now returns `topology_reference_unavailable`. Before, it silently used the sectioned case as the reference, which was the item-2 bug.
- Correct as a fix. But offline teacher audits and decision audits that replay stored states from earlier cells will count these as errors.
- Fix: name the error in the audit summaries. No code change needed.

**L5. The abstention wrapper changes every action's form when on.**
- Where: `psse_env/actions.py:1503-1514`.
- With `abstain_isolated_lines` on, `with_isolated_line_abstentions` returns `normalize_action(action)` for every action, not only the handoff. So the expert's labels change form everywhere.
- The `abstentions` argument is also absent from both tool-schema registries. A student is never told it exists, and an SFT row whose target carries it teaches an argument the schema does not list.
- Off by default, and recorded only.
- Fix: return `action` unchanged for non-handoff actions. Add `abstentions` to the `ask_for_more_evidence` schema before any training run enables the option.

**L6. The declared partner's allowed columns differ between the two collateral paths.**
- Where: `psse_env/oracle/candidate_quality.py:940-941` against `:1050-1054`.
- On the path-valued case (deployment), the partner row may change BR_R, BR_X and BR_B (columns 2, 3, 4). On the mapping case, it may change only `r` and `x`.
- The deployment path with a declared partner has no unit test: `test_pair_refit_collateral.py` uses mapping cases only.
- Fix: limit the path-case partner to columns {2, 3}, and add a path-case test using `matpower_case_differ`.

**L7. Line-ending churn in `candidate_quality.py`.**
- The base file mixed 1,228 CRLF lines with 170 LF-only lines. The head is CRLF throughout.
- Of the 458 changed lines, only 99 insertions and 19 deletions are real. The rest re-ends lines, including the whole unchanged `_hif_meter_only_voltage_nonregression`.
- Harmless, but blame and any merge with a branch that touches that function will show it as rewritten. Fix: none needed. Mention it in the merge commit.

## 2. The six focus areas

### 2.1 `candidate_quality.py` with `pair_refit_adjacent` and `target_first_partial` both off

**Conclusion: no verdict can differ from the pre-stage behaviour for a well-formed candidate.**

**Target-first route** (`:407-413`, `:426-436`):
- `target_first_route` is `False` whenever `self.target_first_partial` is false.
- It is inserted after `branch_first_route` and before `coupled_cluster_route`, so the remaining elif chain is unchanged.
- The provider passes the option through `_deployment_candidate_quality_oracle` with default False, and the research factory passes it with default False. `production_environment_factory` does not pass it.

**Declared partner** (`:878-892`, `:1007-1008`):
- A partner exists only when the candidate's own `metadata.last_parameter_correction` carries `pair_refit_line_index`, and only for `family == "parameter"`.
- That key is written only by the executor when an alternating pass actually ran (`matpower.py:4284-4285`, `refit_done`), which requires `pair_refit_adjacent`.
- Every parameter correction overwrites `last_parameter_correction`, so a stale declaration cannot outlive the correction that wrote it.
- A student cannot write metadata.

**Without a partner** (`extra_rows` empty), the rewritten checks reduce to the old ones:

| Path | Old check | New check | Same result? |
|---|---|---|---|
| Path-valued case | `len(changed_rows) != 1` | `not changed_rows` | Yes |
| Path-valued case | changed row is the target | `target in changed and set(changed) ⊆ {target}` | Yes: changed is exactly {target} |
| Path-valued case | target's columns ⊆ allowed | `all(columns ⊆ allowed)` over one row | Yes |
| Mapping case | single changed row equals the declared row | `declared_index in changed_indices and changed_indices ⊆ {declared}` | Yes |
| Mapping case | `changed_fields != {field}` | `not changed_fields or not changed_fields ⊆ {field}` | Yes: for dicts that differ, `changed_fields` is non-empty |

The only paths that judge a partnerless candidate differently take malformed differ output:
- A `changed_branch_rows` key that is not an integer used to raise `ValueError` out of `int(changed_row)`. It is now caught, and the candidate is scored as collateral, which means REJECT.
- Two keys that collide after `int()` (say `"3"` and `3`) used to fail the length check. They now collapse into one row.

`matpower_case_differ` emits integer keys from `range(...)`, so neither case can arise from the deployment differ.

**Tests that pin this:**
- `test_target_first_partial.py`: the default and branch-first refuse below the floor; the option is off by default in the deployment provider.
- `test_pair_refit_collateral.py`: an undeclared second row is still collateral.
- The existing `test_branch_first_partial.py` fixtures, reused by both.
- The conventional end-to-end in section 3, with identical per-root outcomes under `baseline_compare` at the base and the head.

Gap: no path-case test with a declared partner (L6).

### 2.2 Shared-path changes that run for every teacher

**Planning-reference binding** (`matpower.py:3212-3246` and `:3581-3593`):
- Roots are rendered on the planning case: `scenario_generator.py:2165` and `:4480` render the schematic-normal map. A root therefore takes the `else` branch, and the reference is the current case, as before.
- `planning_reference_case` is recorded by every first `correct_topology`, including a dangling-terminal flip. Only a lineage that has had a structural correction reads it.
- The branch rows correspond one to one: `operator_model_from_map` keeps all `NL` rows, and the code checks the shapes.
- Columns 2-9 are copied from the current case, so an accepted parameter correction survives.
- Status handling (L2) and legacy replays (L4) are the only differences found.

**Evidence boundary** (`psse_env/evidence_profile.py:164-174`, `:198-206`): `planning_reference_case` and `last_parameter_correction` are now admitted into the scada_only and gated *execution* metadata.
- These sets govern what the provider sees after `_evidence_state`, not the policy observation.
- Both keys are written only by executors: `matpower.py:3633` and `:4289`. Neither the scenario generator nor the compound synthesizer writes either key, so neither can carry truth.
- The admission is needed for items 2 and 3 to work under strict profiles at all. Before it, both keys were stripped.
- No leak found here. The leaks are in the WLS ledger (H2, H3), not in the metadata boundary.

**WLS metrics and ledger:**
- `flow_at_noise_lines` is computed for every teacher, on every anomalous non-waveform solve (H3).
- `clean_flip` is computed only when the provider option is on. The factories turn it on exactly when the current variant is the clean-flip teacher: `variant_provider_options` at `release_factories.py:127-137`, `generate_round0_aggregate.py:450-452` and `run_dagger_research.py:320-325`.
- What a student sees: both fields, verbatim, in `fresh_context_evidence.wls`; and `parameter_flow_at_noise_lines` in `fresh_context_evidence.parameter`. Neither is in the history-metric whitelist, so the compacted `last_tool_output` does not repeat them.

**`transactional_env.py`:**
- Two ledger keys (`:859`, `:862`) and one parameter-ledger key (`:4282`).
- The handoff abstention record (`:1692-1698`), written only when the handoff action itself carries `abstentions`.
- Nothing else changed.

### 2.3 Clean-flip cache and planning reference under multiprocessing

**Cache:**
- `_node_breaker_cache` is an instance dict on the provider.
- `evaluate_rollout_suites` calls the environment factory once per episode (`psse_env/dagger/evaluator.py:1611`, `:1961`), and the research factory builds a new `MatpowerDeploymentProviders` on each call (`scripts/run_dagger_research.py:305`).
- So the cache lives for one episode in one process. A forked `ProcessPoolExecutor` worker (`research/compound_errors/rollout.py:354`) or a GPU evaluation worker never shares it.
- Eviction is FIFO at 64 records (`matpower.py:3703-3705`).
- Cached records are only read, never mutated, by the topology context (`_node_breaker_topology_context`, from `:3743`).
- Correct, and bounded.

**Planning reference:** it is a file path in the derived-case directory. A state replayed in another process needs that file, which is true of every derived case already. Nothing new.

**Cost:** see M2.

### 2.4 The default switch

Places that resolve a teacher when none is named:

| Site | Resolves to | Note |
|---|---|---|
| `expert_variants.py:60` (`DEFAULT_EXPERT_VARIANT`) and `:68` (`current_expert_variant`) | The new default | |
| `pipeline.env:199` | The new default | Refused by its own check at `:286` (H1) |
| `release_factories.py:127-137` and `:694` | Current variant (environment variable, else default) | |
| `generate_round0_aggregate.py:450-452`, `:463`, `:2012` | Current variant | |
| `scripts/run_dagger_research.py:320-325`, `:1259`, `:2397` | Current variant | `:1259` reads the environment variable, not `RESEARCH_EXPERT_OPTIONS["variant"]`. Pre-existing. |
| `psse_env/dagger/evaluator.py:848` (offline teacher audit) | Current variant | |
| `research/hpc/full_pipeline_20260907/evaluate_suite.py:141` and `build_suite.py:414`, `:535` | Current variant | |
| `deploy_remote.sh:49` | Accepts all eight variant names | Validates only |
| `prerequisites.sh:75` | Requires `baseline` under classifier-gated | Loud stop |

The CLI tools that take `--expert` keep their own explicit defaults (`research/classifier_triage/expert_e2e.py:157` defaults to `baseline`; the compound scripts default to `baseline_compare`), so they are unaffected.

**Training prompt and SFT export:**
- The tool registries are unconditional (`protocol_bridge.unified_tool_schemas`, `TOOL_JSON_SCHEMAS`). The default therefore does not change the tool list or the system prompt.
- It does change the observations, through `clean_flip` (H2, L1), and the labels: the bounded comparison and the topology-first order.
- A stage reused across the change is refused by the receipt check, because receipts record the variant. That is correct and loud.

**The two recorded override envs:** see M1.

### 2.5 `release_audit.py` and `actions.py`

**Not-applicable rule:**
- `_not_applicable_parameter_targets` (`release_audit.py:553-588`) needs both a parameter fault marked `not_applicable: line_isolated_by_breaker` and a `dangling_line_terminal` topology fault on the same branch row.
- The only writer of the mark is the compound synthesizer: `scenario_generator.py:4665`, inside `_compound_scenario` (`:4446`).
- On a conventional root the set is empty, so every subtraction and `skip_targets` use is the identity.

**New audit checks:**
- `isolated_line_impedance` and `isolated_line_abstention` are added to `checks` with empty `problems`.
- The only consumers of check statuses iterate fixed `required_checks` tuples (`:2680-2696`, `:2819-2832`). Neither tuple contains the new checks.

**Abstentions:**
- `abstention_record` is written only when the handoff carries the argument.
- `with_isolated_line_abstentions` runs only under the expert option, which is off.
- No verdict depends on either.

Neither change can change a verdict on a conventional root.

### 2.6 The four failing tests

All four already fail at the branch base d6e6acc, and so at 6b0209d and 6e5c478. Verified by running them at each of the three commits. None was introduced in the reviewed range.

| Test | Classification | Since | Fix |
|---|---|---|---|
| `psse_env/dagger/test_sft_export.py::NativeToolExportTests::test_full_macro_surface_has_portable_json_schemas` | Real code gap: the controller-protocol registry `TOOL_JSON_SCHEMAS` has no schema for `compare_parameter_candidates`, which `MACRO_ACTIONS` lists. The canonical protocol used by the cells is unaffected. | 3ebc42f | 85c9fa9 |
| `psse_env/examples/test_generate_round0_aggregate.py::GeneratorCliTests::test_cli_exits_nonzero_when_written_artifact_is_not_release_eligible` | Stale test: the CLI stub report lacks `unverified_expert_labels`, which `generate()` has always returned since then | 03af043 | 686eb62 |
| `...::GeneratorCliTests::test_cli_returns_normally_for_release_eligible_artifact` | Stale test, same cause | 03af043 | 686eb62 |
| `...::SystemSwitchTests::test_case57_refuses_ieee14_families_and_waveform_sources` | Stale test: case118 gained the waveform families | 5570bfc | 3171028 |

All three fixes sit on the unmerged branch `claude/five-test-fixes-20261022`: five commits forked at 1193ff3, tip 2654d44.
- Its patch (`git diff 1193ff3 claude/five-test-fixes-20261022`) applies cleanly to 6e5c478.
- With it applied, the four tests and the three other files it touches pass: 117 passed, 3 skipped, 0 failed.

The GeneratorCli fixture fix is a one-line, unambiguous test fix. It was not applied here, because that branch already carries it. Take that branch with this one rather than writing new fixes.

Two more tests failed in the full-suite run (section 3, run 2). Neither is attributable to the branch.

- **`tests/test_suspicion_gated_contract.py::test_the_expert_requests_phasors_on_a_suspicion_and_solves`**
  - It fails alone at both d6e6acc and 6e5c478: the expected tool list ends in `run_wls`, but the expert now finalizes after the estimate.
  - This is a stale test. Commit ece60c9 on the same five-test-fixes branch fixes it, and it passed in the patched-snapshot run above.
- **`tests/test_hif_operating_point_regulation.py::test_injected_hif_candidate_keeps_regulation`**
  - In the full-suite run it failed with "OpenDSS solve did not converge in 3 fresh attempts".
  - It passes alone at d6e6acc and at 6e5c478, and its file passed 7 of 7 twice at 6e5c478.
  - The branch touches neither `three_phase_nlm/` nor the test.
  - It is order-dependent or flaky: the known sporadic OpenDSS divergence.

## 3. What was run, with counts

1. **`code-review` skill, high effort, over d6e6acc..6e5c478.**
   - It reported 10 candidate findings. They are H1, H3, M1, M2 (as a nested-call suspicion, since refuted), L1-L5 and L7 above.
   - H2 and L6 came from the hand pass.
2. **Full test suite at 6e5c478 in this worktree:** `python -m pytest psse_env research tests`, 2,551 tests collected.
   - Result: 2,525 passed, 6 failed, 20 skipped, 544 subtests passed, in 21 minutes. The 6 failures are the four reported ones and the two in section 2.6.
   - The builder's count (1,687 passed, 4 failed) came from a narrower selection. This run adds `tests/`, which holds the two extra failures.
   - This worktree has no gitignored `output/` data, and neither does the builder's. Tests that replay that data skip, so the 20 skips are part of the result. The data-dependent suites were not re-run against the main checkout's `output/`.
3. **The four failing tests**, at d6e6acc, 6b0209d and 6e5c478: 4 failed at each. On a snapshot of 6e5c478 with the five-test-fixes patch applied: 117 passed, 3 skipped, across the five touched test files.
4. **Expert end to end on the conventional 192-root suite** (`development.json`; scada_only, `baseline_compare`, 40 steps, seed 20261006), run at the base snapshot d6e6acc and at 6e5c478 with `research/classifier_triage/expert_e2e.py`.
   - Both runs: 192 of 192 truth-audited successes, 0 false commits, 7 roots with a healthy component touched, about 7.6 minutes.
   - Per root, the tool sequence, the corrections and the outcome are identical on all 192 roots (1,895 steps).
   - This independently confirms the builder's claim for `baseline_compare` on this suite. It says nothing about student observations (H3) or the clean-flip default (H2).
5. **Observation probe** on `development.json`, two roots per family, one `run_wls` each, model-visible preparation as at export.
   - Under the clean-flip default: `clean_flip` present on 4 of 4 topology and measurement+topology roots, absent on 10 of 10 others (H2).
   - Under `baseline`: `flow_at_noise_lines` present on all 14, non-empty on 3 of 4 topology roots (H3).
6. **`pipeline_environment` simulation:** three cases, as given in H1.
7. **Cost probe:** 8 compound roots, every `_node_breaker_estimate` call attributed by call site, as given in M2.

## 4. Merge recommendation

**Merge after named fixes.** Items 1-6 are sound research code:
- the verifier changes are inert when off (2.1);
- the binding repair fixes a real misbinding;
- the release-audit and abstention changes cannot touch a conventional verdict;
- the expert's conventional behaviour is unchanged (section 3, run 4).

What blocks a merge as is: the branch changes shared behaviour for every teacher and student (H3), and its default switch cannot run on the pipeline (H1). In the form the default is used today it also leaks the family on the existing suites (H2).

Fixes required before merging:
1. **H1 and M1:** extend the variant list at `pipeline.env:286`, and pin `EXPERT_VARIANT=baseline` in the two override files that name no teacher.
2. **H3:** keep `flow_at_noise_lines` and `parameter_flow_at_noise_lines` out of the model-visible ledgers unless a run's teacher or verifier option reads them, or version the observation contract in the receipts.
3. **H2 and L1:** run the clean-flip teacher only on suites with uniform node/breaker metadata, and trim the ledger copy of `clean_flip` to its routing fields.
4. **Test suite:** take `claude/five-test-fixes-20261022` with it, so that the suite is green.

Optional: M2, L2-L7.

**Merge target.** The repository's `main` (origin/main, 8b3a29c) is far behind this lineage. The main checkout is on `codex/classifier-triage-20261004` (e7a65e2), one commit past the fork point 1193ff3. Merging this branch there also brings the 14 base commits between 1193ff3 and d6e6acc (the conventional-only cell's follow-ups).

**Sibling branches.** `claude/agitated-sammet-93f94e` and `claude/pensive-darwin-33a0d8` fork inside the reviewed range at a52a7e5, before items 6 and the default switch. Any merge order should account for them.

**If the fixes are not wanted now**, keep this as a research branch and do not merge it into the line the paper and the earlier cells were built on.
