#!/usr/bin/env bash
# Where the minimal environment's supervised pass stands.  Read-only; run on torch or through ssh stdin:
#
#   wsl -- ssh torch bash -s < research/hpc/minimal_sft_20261115/status_cell.sh
exec 2>&1
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_sft_20261115
OUT=$WORK/out
LOGS=$WORK/logs
echo "at $(date -u +%FT%TZ) on $(hostname)"
echo "=== source and rows"
git -C "$WORK/source" rev-parse HEAD 2>/dev/null | sed 's/^/    source /'
ls -la "$WORK/data/rows" 2>/dev/null | tail -n +2 | awk '{print "    " $5, $9}'
echo "=== submitted"
tail -n 12 "$WORK/submitted_jobs.txt" 2>/dev/null | sed 's/^/    /'
echo "=== queue"
squeue -u "$USER" -h -o "%i %j %T %M %R" | grep -E "msf-" | sed 's/^/    /'
[ -s "$OUT/row_gate.json" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print("=== row gate:", d.get("rows_gated"), "rows in", d.get("gate_seconds"), "s; lengths", json.dumps(d.get("length_audit"))[:700]); print("    actions", json.dumps(d.get("action_distribution")))' "$OUT/row_gate.json"
for stage in smoke_base smoke_r1 pass_base pass_r1; do
  dir=$OUT/$stage
  [ -d "$dir/training" ] || continue
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
report = f"{stage}/training/training_stage.json"
if os.path.isfile(report):
    # How the forward was called: "kept" got the target's span alone; "ignored" or "refused" means the memory was not saved.
    print("    supervised-span loss:", json.dumps(json.load(open(report)).get("supervised_logits"), sort_keys=True))
receipt = f"{stage}/training.done"
if os.path.isfile(receipt):
    done = json.load(open(receipt))
    print("    receipt:", json.dumps({key: done.get(key) for key in ("mode", "start", "learning_rate", "train_rows", "max_steps", "global_step", "research_run_passed",
                                                                    "validation_losses", "adapter_tree_sha256")})[:900])
PY
done
echo "=== evaluations in the minimal environment (300 development cases: the rule expert's recorded episodes repair 184, truth-built sequences 247)"
python3 - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]
KEYS = ("cases", "solved", "strict_solved", "solved_and_declared_resolved", "solved_but_not_declared", "false_resolved", "declared_unresolved",
        "episodes_with_a_damaged_healthy_object", "tool_errors", "budget_exhausted", "prompt_limit", "estimator_passes_at_end")
for label, name in (("the pass from the base model", "eval_dev300_base"), ("the pass from R1", "eval_dev300_r1"), ("smoke: R1 on eight cases", "eval_smoke")):
    folder = f"{out}/{name}"
    if not os.path.isdir(folder):
        continue
    episodes = f"{folder}/episodes"
    print(f"    {label}: episodes written {len(os.listdir(episodes)) if os.path.isdir(episodes) else 0}")
    path = f"{folder}/evaluation.json"
    if not os.path.isfile(path):
        print("        evaluation.json: not yet")
        continue
    summary = json.load(open(path)).get("summary") or {}
    totals = summary.get("totals") or {}
    print("        " + ", ".join(f"{key} {totals.get(key, 0)}" for key in KEYS))
    print(f"        mean actions {summary.get('mean_actions')}, mean reward {summary.get('mean_reward')}, prompt tokens {json.dumps(summary.get('prompt_tokens'))}")
    print(f"        actions by tool {json.dumps(summary.get('actions_by_tool'), sort_keys=True)}")
    for group, cell in (summary.get("by_reference") or {}).items():
        print(f"        {group}: solved {cell.get('solved', 0)} of {cell.get('cases', 0)} (strict {cell.get('strict_solved', 0)})")
    errors = {key[6:]: value for key, value in totals.items() if key.startswith("error:")}
    if errors:
        print(f"        refused calls by code {json.dumps(errors, sort_keys=True)}")
PY
echo "=== logs: preemptions, and lines that name a failure"
for log in "$LOGS"/msf-*.out; do
  [ -f "$log" ] || continue
  echo "    $(basename "$log"): $(wc -l < "$log") lines; preemptions $(grep -c 'DUE TO PREEMPTION' "$log"); 'Traceback' $(grep -c 'Traceback' "$log"); last: $(grep -vE 'Warning|warnings.warn|^\s*$' "$log" | tail -1 | cut -c1-200)"
done
echo status_done
