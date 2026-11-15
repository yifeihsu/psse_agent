#!/usr/bin/env bash
# Where the complete-evidence clone stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/clone_complete_evidence_20261115/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/clone_complete_evidence_20261115
OUT=$WORK/out
LOGS=$WORK/logs
ARM_A_BASELINE=/scratch/yx3882/rl_improvement_20261110/out/arm_a_bounded/baseline/evaluation.json
OPEN_CELL_OUT=/scratch/yx3882/clone_open_rules_20261112/out
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source and rows"
git -C "$WORK/source" rev-parse HEAD 2>/dev/null | sed 's/^/    source /'
ls -la "$WORK/data/rows" 2>/dev/null | tail -n +2 | awk '{print "    " $5, $9}'
echo "=== submitted"
tail -n 10 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "cec-" | sed 's/^/    /'
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
echo "=== baselines (the first three are read in place: S0 with the rules in place, then S0 and the open-rules clone with them lifted)"
python3 - "$OUT" "$ARM_A_BASELINE" "$OPEN_CELL_OUT" <<'PY'
import json, os, sys
out, arm_a, open_out = sys.argv[1], sys.argv[2], sys.argv[3]
KEYS = ("truth_audited_task_success", "clean_success", "mean_steps", "false_commit_count", "invalid_action_count", "loop_episodes",
        "healthy_component_corrupted", "prompt_limit_refusals", "solved_within_actions", "lifecycle", "open_rules")
runs = [("arm A (S0, rules in place, old prompt)", os.path.dirname(arm_a), arm_a)]
for label, name in (("S0, rules lifted, old prompt", "s0_open"), ("the open-rules clone, rules lifted, old prompt", "clone_open")):
    run = f"{open_out}/baseline_{name}"
    runs.append((label, f"{run}/baseline", f"{run}/baseline/evaluation.json"))
for label, name in (("this cell's clone, rules lifted, complete evidence", "clone_complete"), ("smoke: S0 on eight cases, complete evidence", "smoke_complete")):
    run = f"{out}/baseline_{name}"
    runs.append((label, f"{run}/baseline", f"{run}/baseline/evaluation.json"))
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
echo "=== logs: preemptions, and lines that name a failure"
for log in "$LOGS"/cec-*.out; do
  [ -f "$log" ] || continue
  echo "    $(basename "$log"): $(wc -l < "$log") lines; preemptions $(grep -c 'DUE TO PREEMPTION' "$log"); 'Traceback' $(grep -c 'Traceback' "$log"); last: $(grep -vE 'Warning|warnings.warn|^\s*$' "$log" | tail -1 | cut -c1-200)"
done
echo status_done
