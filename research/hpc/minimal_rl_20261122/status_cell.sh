#!/usr/bin/env bash
# Where the runs and the probe of the RL plan of 2026-11-22 stand.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/minimal_rl_20261122/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_rl_20261122
OUT=$WORK/out
LOGS=$WORK/logs
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source"
git -C "$WORK/source" rev-parse HEAD 2>/dev/null | sed 's/^/    source /'
echo "=== the supervised pass every run starts from"
[ -s "$PASS/training.done" ] && echo "    receipt present; adapter $PASS/training/lora" || echo "    no receipt (the pass has not finished)"
echo "=== submitted"
tail -n 12 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "mrp-" | sed 's/^/    /'
for probe in smoke dev train; do
  dir=$OUT/probe_$probe
  [ -d "$dir" ] || continue
  echo "=== probe $probe: one-step forks of the supervised model's greedy episodes, no update"
  python3 - "$dir" <<'PROBEPY'
import glob, json, os, sys

folder = sys.argv[1]
print(f"    cases forked {len(glob.glob(f'{folder}/cases/*.json'))}")
path = f"{folder}/summary.json"
if not os.path.isfile(path):
    print("    summary.json: not yet")
else:
    payload = json.load(open(path))
    summary = payload.get("summary") or {}
    totals = summary.get("totals") or {}
    unrepaired = int(totals.get("own_episode_unrepaired", 0))
    print(f"    selected {((payload.get('selection') or {}).get('selected'))}; greedy episode unrepaired {unrepaired} of {totals.get('cases', 0)}; "
          f"one forced step repairs {totals.get('unrepaired_cases_one_forced_step_repairs', 0)} of them "
          f"({totals.get('unrepaired_cases_one_forced_step_repairs_in_the_policys_own_decoding', 0)} with the model's own decoding of the arguments)")
    statuses = {key[9:]: value for key, value in totals.items() if key.startswith("branches_")}
    print(f"    states {totals.get('states', 0)}, branches {totals.get('branches', 0)} {json.dumps(statuses, sort_keys=True)}; "
          f"forced calls the environment refused {totals.get('forced_calls_the_environment_refused', 0)}")
    print(f"    forced prefixes that start the call's own rendering: {totals.get('forced_prefixes_canonical', 0)} of "
          f"{totals.get('forced_prefixes_checked', 0)} checked")
    print(f"    states with a better branch {totals.get('states_with_a_better_branch', 0)} in {totals.get('cases_with_a_better_branch', 0)} cases; "
          f"by tool {json.dumps(summary.get('better_branches_by_tool'), sort_keys=True)}; gain {json.dumps(summary.get('gain'), sort_keys=True)}")
    print(f"    first repairing branch by tool {json.dumps(summary.get('first_repairing_branch_by_tool'), sort_keys=True)}; "
          f"better branches at the last state {totals.get('better_branches_at_the_last_state', 0)}")
    for name, cell in (summary.get("by_class") or {}).items():
        print(f"        {name}: unrepaired {cell.get('own_episode_unrepaired', 0)} of {cell.get('cases', 0)}, "
              f"one forced step repairs {cell.get('unrepaired_cases_one_forced_step_repairs', 0)}")
    cases = max(int(totals.get("cases", 0)), 1)
    print(f"    fork time {round(float(summary.get('seconds') or 0) / 3600, 2)} h in all, {round(float(summary.get('seconds') or 0) / cases, 1)} s a case")
PROBEPY
done
for run in steps_smoke steps forks_smoke forks; do
  dir=$OUT/$run
  [ -s "$dir/manifest.json" ] || continue
  echo "=== $run"
  python3 - "$dir" <<'PY'
import glob, json, os, sys

run = sys.argv[1]


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def evaluation_line(label, payload):
    if not payload:
        return
    summary = payload.get("summary") or {}
    if "skipped" in summary:
        print(f"    {label}: skipped ({summary['skipped']})")
        return
    totals = summary.get("totals") or {}
    keys = ("cases", "solved", "strict_solved", "solved_and_declared_resolved", "false_resolved", "declared_unresolved",
            "budget_exhausted", "prompt_limit", "episodes_with_a_damaged_healthy_object", "tool_errors")
    print(f"    {label}: " + ", ".join(f"{key} {totals.get(key, 0)}" for key in keys)
          + f"; mean actions {summary.get('mean_actions')}, mean reward {summary.get('mean_reward')}")
    solvable = [cell for group, cell in (summary.get("by_reference") or {}).items() if group != "neither repairs"]
    if solvable:
        print(f"        of the cases a truth-built sequence repairs: solved {sum(cell.get('solved', 0) for cell in solvable)} of "
              f"{sum(cell.get('cases', 0) for cell in solvable)}")
    for group, cell in (summary.get("by_reference") or {}).items():
        print(f"        {group}: solved {cell.get('solved', 0)} of {cell.get('cases', 0)}")
    against = payload.get("against_baseline")
    if against:
        print(f"        against the baseline: {against.get('solved_before')} -> {against.get('solved_now')} of "
              f"{against.get('cases_in_both')}, gained {len(against.get('gained') or [])}, lost {len(against.get('lost') or [])}")
    if payload.get("baseline_source"):
        print(f"        source: {payload['baseline_source']}")


manifest = load(f"{run}/manifest.json") or {}
print(f"    at iteration {manifest.get('iteration')}, phase {manifest.get('phase')}; adapter {manifest.get('current_adapter')}")
evaluation_line("baseline", load(f"{run}/baseline/evaluation.json"))
done = len(glob.glob(f"{run}/baseline/evaluation/episodes/*.json"))
if done and not os.path.isfile(f"{run}/baseline/evaluation.json"):
    print(f"    baseline: {done} episodes written, no summary yet")
for folder in sorted(glob.glob(f"{run}/iteration_*")):
    print(f"    --- {os.path.basename(folder)}")
    groups = len(glob.glob(f"{folder}/trajectories/groups/*.jsonl"))
    forked = len(glob.glob(f"{folder}/forks/cases/*.json"))
    rollout = load(f"{folder}/rollout_summary.json")
    if rollout:
        store = rollout.get("store") or {}
        totals = store.get("totals") or {}
        keys = ("groups", "episodes", "solved", "false_resolved", "declared_unresolved", "budget_exhausted", "prompt_limit",
                "unpaired_trajectories", "groups_with_reward_spread", "groups_with_repaired_and_unrepaired_samples",
                "groups_with_every_sample_repaired", "groups_with_no_sample_repaired")
        print("        rollout: " + ", ".join(f"{key} {totals.get(key, 0)}" for key in keys))
        print(f"        rollout: mean reward {store.get('mean_reward')}, mean actions {store.get('mean_actions')}, steps with every "
              f"token above 0.99: {store.get('fraction_of_steps_with_every_token_above_0_99')} of {store.get('sampled_steps_with_probabilities')}; "
              f"{round(float(rollout.get('seconds') or 0) / 3600, 2)} h in the closing call")
        focus = rollout.get("focus") or {}
        for name in ("left_unrepaired_by_the_starting_adapter", "repaired_by_the_starting_adapter"):
            cell = focus.get(name)
            if cell:
                print(f"        focus, cases {name.replace('_', ' ')}: {cell.get('cases')} drawn of {(focus.get('available') or {}).get(name)}; "
                      f"with a repaired sample {cell.get('cases_with_a_repaired_sample')}, with none {cell.get('cases_with_no_sample_repaired')}; "
                      f"episodes repaired {cell.get('episodes_repaired')} of {cell.get('episodes')}")
        examples = rollout.get("truth_built_examples")
        if examples:
            print(f"        truth-built examples: {examples.get('examples')} kept for {examples.get('cases_without_a_repaired_sample')} cases "
                  f"without a repaired sample")
        forks = rollout.get("forks")
        if forks:
            ftotals = forks.get("totals") or {}
            statuses = {key[9:]: value for key, value in ftotals.items() if key.startswith("branches_")}
            print(f"        forks: {forks.get('cases_forked')} of {forks.get('cases_drawn')} cases; greedy episode unrepaired "
                  f"{ftotals.get('own_episode_unrepaired', 0)}, one forced step repairs {ftotals.get('unrepaired_cases_one_forced_step_repairs', 0)} of them; "
                  f"states {ftotals.get('states', 0)}, branches {ftotals.get('branches', 0)} {json.dumps(statuses, sort_keys=True)}")
            print(f"        forks: states with a better branch {ftotals.get('states_with_a_better_branch', 0)}, by tool "
                  f"{json.dumps(forks.get('better_branches_by_tool'), sort_keys=True)}, gain {json.dumps(forks.get('gain'), sort_keys=True)}; "
                  f"forced prefixes canonical {ftotals.get('forced_prefixes_canonical', 0)} of {ftotals.get('forced_prefixes_checked', 0)}; "
                  f"{round(float(forks.get('seconds') or 0) / 3600, 2)} h of forks")
    elif groups or forked:
        print(f"        rollout: {groups} groups and {forked} forked cases written, not closed")
    report = load(f"{folder}/update_report.json")
    if report:
        rows = report.get("rows") or {}
        keys = ("rows", "policy_rows", "preservation_rows", "informative_groups", "groups_incomplete", "trajectories_unpaired",
                "prompt_length_mismatches", "prompt_digest_mismatches", "prompt_digest_missing", "prompt_truncated_rows",
                "rows_without_sampling_record_rejected", "supervised_rows", "supervised_rows_unrenderable")
        print("        rows: " + ", ".join(f"{key} {rows.get(key, 0)}" for key in keys))
        update = report.get("update")
        taught = rows.get("truth_built_examples")
        if taught:
            print(f"        truth-built examples in the update: {taught.get('rows')} rows of {taught.get('examples_with_rows')} examples, "
                  f"row weights {json.dumps(taught.get('row_weights'))}")
        forks = rows.get("one_step_forks")
        if forks:
            print(f"        fork rows in the update: {forks.get('rows')} of {forks.get('candidates')} better branches in {forks.get('cases')} cases, "
                  f"by tool {json.dumps(forks.get('rows_by_tool'), sort_keys=True)}; whose branch repairs {forks.get('rows_whose_branch_repairs')}, "
                  f"within one action's cost {forks.get('rows_within_one_action_cost')}; row weights {json.dumps(forks.get('row_weights'))}; "
                  f"not rendered {forks.get('unrenderable')}, over the input limit {forks.get('prompt_over_the_input_limit')}")
        if update is None:
            print(f"        update: none ({report.get('note')})")
        else:
            checks = update.get("on_policy_check") or [{}]
            print(f"        update: {update.get('optimizer_steps')} step(s) on {update.get('rows')} rows in {round(float(update.get('seconds') or 0) / 3600, 2)} h; "
                  f"rows per step {(update.get('row_accounting') or {}).get('rows_per_step')}")
            print(f"        update: policy loss {update.get('policy_loss')}, kl {update.get('kl')}, supervised loss {update.get('supervised_loss')}, "
                  f"grad norm {update.get('grad_norm')}")
            # How far each step's log-probabilities are from the sampled ones: the first is the numerical gap, the
            # later ones add what the earlier steps moved.
            print("        update: mean |log ratio| to the sampler by step "
                  + json.dumps([check.get("mean_abs_log_ratio") for check in checks])
                  + ", on uncertain tokens " + json.dumps([check.get("uncertain_mean_abs_log_ratio") for check in checks]))
            print(f"        update: optimizer {json.dumps(report.get('optimizer'), sort_keys=True)}; resumed from {update.get('resumed_from')}")
        integrity = report.get("integrity") or {}
        print(f"        integrity: ok {integrity.get('ok')}; failures {integrity.get('failures')}; warnings {integrity.get('warnings')}")
    elif os.path.isdir(f"{folder}/update_checkpoint"):
        state = load(f"{folder}/update_checkpoint/state.json") or {}
        print(f"        update: running; checkpoint at step {state.get('completed_chunks')}, row {state.get('chunk_rows_done')}")
    if os.path.isfile(f"{folder}/integrity_failure.json"):
        print(f"        STOPPED: {json.dumps(load(f'{folder}/integrity_failure.json'))[:600]}")
    evaluation_line("evaluation", load(f"{folder}/evaluation.json"))
    done = len(glob.glob(f"{folder}/evaluation/episodes/*.json"))
    if done and not os.path.isfile(f"{folder}/evaluation.json"):
        print(f"        evaluation: {done} episodes written, no summary yet")
PY
done
echo "=== logs: preemptions, lines that name a failure, and the card"
for log in "$LOGS"/mrp-*.out; do
  [ -f "$log" ] || continue
  echo "    $(basename "$log"): $(wc -l < "$log") lines; preemptions $(grep -c 'DUE TO PREEMPTION' "$log"); 'Traceback' $(grep -c 'Traceback' "$log"); last: $(grep -vE 'Warning|warnings.warn|^\s*$' "$log" | tail -1 | cut -c1-200)"
done
for samples in "$LOGS"/gpu-util-rl-*.csv "$LOGS"/gpu-util-probe-*.csv; do
  [ -f "$samples" ] || continue
  awk -F', ' -v name="$(basename "$samples")" '{gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0; gsub(/ %/,"",$2); s+=$2; n++} END {if (n) printf "    %s: %d samples, peak memory %d MiB, mean utilization %.0f%%\n", name, n, m, s/n}' "$samples"
done
echo status_done
