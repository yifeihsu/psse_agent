#!/usr/bin/env bash
# Where the minimal environment's RL run stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/minimal_rl_20261116/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_rl_20261116
OUT=$WORK/out
LOGS=$WORK/logs
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source"
git -C "$WORK/source" rev-parse HEAD 2>/dev/null | sed 's/^/    source /'
echo "=== the supervised pass this run starts from"
[ -s "$PASS/training.done" ] && echo "    receipt present; adapter $PASS/training/lora" || echo "    no receipt yet (the pass has not finished)"
[ -s "$WORK/data/smoke_adapter/SOURCE.txt" ] && sed 's/^/    smoke adapter: /' "$WORK/data/smoke_adapter/SOURCE.txt"
echo "=== submitted"
tail -n 12 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "mrl-" | sed 's/^/    /'
for run in smoke ieee14; do
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
    rollout = load(f"{folder}/rollout_summary.json")
    if rollout:
        store = rollout.get("store") or {}
        totals = store.get("totals") or {}
        keys = ("groups", "episodes", "solved", "false_resolved", "declared_unresolved", "budget_exhausted", "prompt_limit",
                "unpaired_trajectories", "groups_with_reward_spread", "groups_with_distinct_samples",
                "groups_with_repaired_and_unrepaired_samples", "groups_with_every_sample_repaired", "groups_with_no_sample_repaired")
        print("        rollout: " + ", ".join(f"{key} {totals.get(key, 0)}" for key in keys))
        print(f"        rollout: mean reward {store.get('mean_reward')}, mean actions {store.get('mean_actions')}, steps with every "
              f"token above 0.99: {store.get('fraction_of_steps_with_every_token_above_0_99')} of {store.get('sampled_steps_with_probabilities')}")
        errors = {key[6:]: value for key, value in totals.items() if key.startswith("error:")}
        if errors:
            print(f"        rollout: refused calls by code {json.dumps(errors, sort_keys=True)}")
    elif groups:
        print(f"        rollout: {groups} groups written, not closed")
    report = load(f"{folder}/update_report.json")
    if report:
        rows = report.get("rows") or {}
        keys = ("rows", "policy_rows", "preservation_rows", "informative_groups", "groups_incomplete", "trajectories_unpaired",
                "prompt_length_mismatches", "prompt_digest_mismatches", "prompt_digest_missing", "prompt_truncated_rows",
                "rows_without_sampling_record_rejected")
        print("        rows: " + ", ".join(f"{key} {rows.get(key, 0)}" for key in keys))
        update = report.get("update")
        if update is None:
            print(f"        update: none ({report.get('note')})")
        else:
            check = (update.get("on_policy_check") or [{}])[0]
            print(f"        update: {update.get('optimizer_steps')} step(s) on {update.get('rows')} rows in {round(float(update.get('seconds') or 0) / 3600, 2)} h; "
                  f"policy loss {update.get('policy_loss')}, kl {update.get('kl')}, grad norm {update.get('grad_norm')}")
            print(f"        update: first step against the sampler: mean |log ratio| {check.get('mean_abs_log_ratio')}, on uncertain tokens "
                  f"{check.get('uncertain_mean_abs_log_ratio')} ({check.get('uncertain_tokens')} tokens), largest {check.get('max_abs_log_ratio')}, "
                  f"rows without a record {check.get('rows_without_record')}")
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
for log in "$LOGS"/mrl-*.out; do
  [ -f "$log" ] || continue
  echo "    $(basename "$log"): $(wc -l < "$log") lines; preemptions $(grep -c 'DUE TO PREEMPTION' "$log"); 'Traceback' $(grep -c 'Traceback' "$log"); last: $(grep -vE 'Warning|warnings.warn|^\s*$' "$log" | tail -1 | cut -c1-200)"
done
for samples in "$LOGS"/gpu-util-rl-*.csv; do
  [ -f "$samples" ] || continue
  awk -F', ' -v name="$(basename "$samples")" '{gsub(/ MiB/,"",$3); if ($3+0>m) m=$3+0; gsub(/ %/,"",$2); s+=$2; n++} END {if (n) printf "    %s: %d samples, peak memory %d MiB, mean utilization %.0f%%\n", name, n, m, s/n}' "$samples"
done
echo status_done
