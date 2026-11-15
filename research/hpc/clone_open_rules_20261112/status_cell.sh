#!/usr/bin/env bash
# Where the open-rules clone stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/clone_open_rules_20261112/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/clone_open_rules_20261112
OUT=$WORK/out
LOGS=$WORK/logs
ARM_A_BASELINE=/scratch/yx3882/rl_improvement_20261110/out/arm_a_bounded/baseline/evaluation.json
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source and rows"
git -C "$WORK/source" rev-parse HEAD 2>/dev/null | sed 's/^/    source /'
ls -la "$WORK/data/rows" 2>/dev/null | tail -n +2 | awk '{print "    " $5, $9}'
echo "=== submitted"
tail -n 10 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "orc-" | sed 's/^/    /'
[ -s "$OUT/clone/row_gate.json" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("=== row gate:", d.get("rows_gated"), "rows in", d.get("gate_seconds"), "s; lengths", json.dumps(d.get("length_audit"))[:600])' "$OUT/clone/row_gate.json"
for stage in smoke clone; do
  dir=$OUT/$stage
  [ -d "$dir" ] || continue
  echo "=== $stage"
  [ -s "$dir/training.done" ] && echo "    receipt present" || echo "    no receipt yet"
  [ -s "$dir/gate_timing.json" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("    row gate:", d.get("rows_gated"), "rows in", d.get("gate_seconds"), "s; lengths", json.dumps(d.get("length_audit"))[:500])' "$dir/gate_timing.json"
  ls -d "$dir"/training/checkpoint-* 2>/dev/null | sed 's#.*/checkpoint-##' | sort -n | tail -3 | tr '\n' ' ' | sed 's/^/    newest checkpoints: /'; echo
  python3 - "$dir" <<'PY'
import glob, json, os, sys
stage = sys.argv[1]
states = sorted(glob.glob(f"{stage}/training/checkpoint-*/trainer_state.json"), key=lambda path: int(path.split("checkpoint-")[1].split("/")[0]))
for path in ([f"{stage}/training/trainer_state.json"] if os.path.isfile(f"{stage}/training/trainer_state.json") else states[-1:]):
    state = json.load(open(path))
    history = state.get("log_history") or []
    losses = [entry["loss"] for entry in history if "loss" in entry]
    evaluations = [(entry.get("step"), round(entry["eval_loss"], 4)) for entry in history if "eval_loss" in entry]
    recent = losses[-64:]
    print(f"    step {state.get('global_step')} of {state.get('max_steps')}; training loss, last {len(recent)} steps: "
          f"{sum(recent) / max(1, len(recent)):.4f}; validation losses {evaluations[-6:]}")
receipt = f"{stage}/training.done"
if os.path.isfile(receipt):
    done = json.load(open(receipt))
    print("    receipt:", json.dumps({key: done.get(key) for key in ("mode", "train_rows", "max_steps", "global_step", "research_run_passed",
                                                                    "validation_losses", "adapter_tree_sha256")})[:900])
PY
done
echo "=== baselines (arm A's is S0 with the rules in place)"
python3 - "$OUT" "$ARM_A_BASELINE" <<'PY'
import json, os, sys
out, arm_a = sys.argv[1], sys.argv[2]
KEYS = ("truth_audited_task_success", "clean_success", "mean_steps", "false_commit_count", "invalid_action_count", "loop_episodes",
        "healthy_component_corrupted", "prompt_limit_refusals", "solved_within_actions", "lifecycle", "open_rules")
runs = [("arm A (S0, rules in place)", os.path.dirname(arm_a), arm_a)]
for name in ("s0_open", "clone_open", "smoke_open"):
    run = f"{out}/baseline_{name}"
    runs.append((name, f"{run}/baseline", f"{run}/baseline/evaluation.json"))
for label, folder, path in runs:
    if not os.path.isdir(folder):
        continue
    episodes = f"{folder}/evaluation/episodes"
    print(f"    {label}: episodes written {len(os.listdir(episodes)) if os.path.isdir(episodes) else 0}")
    if os.path.isfile(path):
        summary = json.load(open(path)).get("summary") or {}
        for key in KEYS:
            if key in summary:
                print(f"        {key}: {json.dumps(summary.get(key), sort_keys=True)}")
    else:
        print("        evaluation.json: not yet")
PY
echo "=== the RL arm (the run the clone's baseline began)"
python3 - "$OUT/baseline_clone_open" <<'PY'
import glob, json, os, sys
run = sys.argv[1]
manifest = f"{run}/manifest.json"
if not os.path.isfile(manifest):
    print("    no manifest yet")
    raise SystemExit(0)
state = json.load(open(manifest))
print(f"    manifest: iteration {state.get('iteration')} phase {state.get('phase')} adapter {state.get('current_adapter')}")
for folder in sorted(glob.glob(f"{run}/iteration_*")):
    name = os.path.basename(folder)
    groups = len(glob.glob(f"{folder}/trajectories/groups/*.jsonl"))
    episodes = len(glob.glob(f"{folder}/evaluation/episodes/*.json"))
    files = [item for item in ("rollout_summary.json", "decision_values.json", "update_report.json", "integrity_failure.json",
                               "evaluation.json") if os.path.isfile(f"{folder}/{item}")]
    print(f"    {name}: rollout groups {groups}, evaluation episodes {episodes}, files {files}")
    checkpoint = f"{folder}/update_checkpoint/state.json"
    if os.path.isfile(checkpoint) and "update_report.json" not in files:
        saved = json.load(open(checkpoint))
        print("        update checkpoint:", json.dumps({key: saved.get(key) for key in ("row_count", "planned_steps", "chunk_index",
                                                                                      "chunk_rows_done", "steps_done")}))
    if "rollout_summary.json" in files:
        store = (json.load(open(f"{folder}/rollout_summary.json")).get("store") or {})
        print("        rollout:", json.dumps({key: store.get(key) for key in ("episodes", "success_rate", "mean_calls",
                                                                              "groups_with_reward_spread", "lifecycle",
                                                                              "rejected_candidates")}, sort_keys=True)[:900])
    if "update_report.json" in files:
        report = json.load(open(f"{folder}/update_report.json"))
        integrity = report.get("integrity") or {}
        rows = report.get("rows") or {}
        print("        update:", json.dumps({"integrity_ok": integrity.get("ok"), "failures": integrity.get("failures"),
                                             "rows": rows.get("rows"), "supervised_rows": rows.get("supervised_rows"),
                                             "supervised_rows_on_rejected_candidates": rows.get("supervised_rows_on_rejected_candidates"),
                                             "target_renderer": rows.get("target_renderer")}, sort_keys=True)[:900])
    if "evaluation.json" in files:
        summary = json.load(open(f"{folder}/evaluation.json")).get("summary") or {}
        print("        evaluation:", json.dumps({key: summary.get(key) for key in ("truth_audited_task_success", "clean_success",
                                                                                   "healthy_component_corrupted", "invalid_action_count",
                                                                                   "loop_episodes", "mean_steps", "lifecycle")},
                                                 sort_keys=True)[:900])
PY
echo "=== logs: preemptions, and lines that name a failure"
for log in "$LOGS"/orc-*.out; do
  [ -f "$log" ] || continue
  echo "    $(basename "$log"): $(wc -l < "$log") lines; preemptions $(grep -c 'DUE TO PREEMPTION' "$log"); 'Traceback' $(grep -c 'Traceback' "$log"); last: $(grep -vE 'Warning|warnings.warn|^\s*$' "$log" | tail -1 | cut -c1-200)"
done
echo status_done
