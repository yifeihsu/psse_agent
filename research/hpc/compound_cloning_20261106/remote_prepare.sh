#!/usr/bin/env bash
# The cluster side of deploy_from_windows.sh, sent through ssh stdin once the source, rows and suites
# are in place:
#
#   bash -s -- COMMIT BRANCH [TRAIN_SHA256 VALIDATION_SHA256] < remote_prepare.sh
#
# Copies the start adapter into the cell once, writes the rows manifest and the deploy record, and
# checks what can only be checked on torch.  Prints every problem it finds, then fails if there was one.
set -euo pipefail
COMMIT=${1:?commit}
BRANCH=${2:?branch}
TRAIN_SHA=${3:-}
VALIDATION_SHA=${4:-}
WORK=/scratch/yx3882/compound_cloning_20261106
# shellcheck source=cell.env
source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"
cell_environment
PROBLEMS=0
problem() { echo "PROBLEM: $*" >&2; PROBLEMS=1; }

for script in "$CELL"/*.sh "$CELL"/*.sbatch "$CELL"/*.env "$DATA/eval_suites.env"; do
  bash -n "$script" || problem "syntax: $script"
done
[[ -x "$PY" ]] || problem "python missing: $PY"

# A copy, not a link: checkpoint_tree_sha256 refuses symlinked and multiply linked files (hard-linked
# views of that cell's checkpoints failed this way on 2026-11-01), and the source cell may change.
if [[ ! -s "$START_ADAPTER/adapter_model.safetensors" ]]; then
  if [[ -s "$START_ADAPTER_SOURCE/adapter_model.safetensors" && -s "$START_ADAPTER_SOURCE/adapter_config.json" ]]; then
    rm -rf "$START_ADAPTER.part"
    cp -r "$START_ADAPTER_SOURCE" "$START_ADAPTER.part"
    mv "$START_ADAPTER.part" "$START_ADAPTER"
    echo "start adapter copied from $START_ADAPTER_SOURCE"
  else
    problem "start adapter not found at $START_ADAPTER_SOURCE (expected adapter_model.safetensors and adapter_config.json)"
  fi
else
  echo "start adapter already in place: $START_ADAPTER"
fi

[[ -s "$CONVENTIONAL_TEST_SUITE" ]] || problem "conventional test suite not found: $CONVENTIONAL_TEST_SUITE"

"$PY" - "$COMMIT" "$BRANCH" "$TRAIN_VIEW" "$VALIDATION_VIEW" "$TRAIN_SHA" "$VALIDATION_SHA" "$ROWS_MANIFEST" "$START_ADAPTER" "$START_ADAPTER_SOURCE" "$WORK/deploy.json" "$HF_HOME" "$CONVENTIONAL_TEST_SUITE" <<'PY' || PROBLEMS=1
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

commit, branch, train, validation, train_sha, validation_sha, manifest, adapter, adapter_source, deploy, hf_home, test_suite = sys.argv[1:13]
problems = []
packages = {}
for name in ("torch", "transformers", "trl", "peft", "datasets", "bitsandbytes"):
    try:
        packages[name] = version(name)
    except PackageNotFoundError:
        problems.append(f"python package missing: {name}")
from psse_env.research_models import get_research_model_spec
spec = get_research_model_spec("12b")
snapshot = Path(hf_home) / "hub" / ("models--" + spec.model_id.replace("/", "--")) / "snapshots" / spec.revision
if not snapshot.is_dir():
    problems.append(f"model snapshot not cached: {snapshot}")
from psse_env.dagger.release_factories import checkpoint_tree_sha256
adapter_tree = None
if Path(adapter, "adapter_model.safetensors").is_file():
    try:
        adapter_tree = checkpoint_tree_sha256(adapter)
    except ValueError as exc:
        problems.append(f"start adapter refused by the tree hash: {exc}")
    recorded = json.loads(Path(adapter, "adapter_config.json").read_text(encoding="utf-8")).get("base_model_name_or_path")
    if recorded != spec.model_id:
        problems.append(f"start adapter base model {recorded!r} is not {spec.model_id!r}: research_checkpoint would refuse every checkpoint")
rows = None
if train_sha and Path(train).is_file():
    with open(train, encoding="utf-8") as stream:
        count = sum(1 for line in stream if line.strip())
    with open(validation, encoding="utf-8") as stream:
        validation_count = sum(1 for line in stream if line.strip())
    rows = {"train_view": train, "train_sha256": train_sha, "train_rows": count, "optimizer_steps": math.ceil(count / 4),
            "validation_view": validation, "validation_sha256": validation_sha, "validation_rows": validation_count}
    Path(manifest).write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
conventional_roots = None
if Path(test_suite).is_file():
    conventional_roots = len(json.loads(Path(test_suite).read_text(encoding="utf-8")))
    if conventional_roots != 192:
        problems.append(f"conventional test suite has {conventional_roots} roots, not 192: {test_suite}")
record = {"contract": "compound_cloning_deploy_v1", "source_commit": commit, "branch": branch, "packages": packages,
          "model_snapshot": str(snapshot), "start_adapter": adapter, "start_adapter_source": adapter_source,
          "start_adapter_tree_sha256": adapter_tree, "rows": rows, "conventional_test_roots": conventional_roots}
Path(deploy).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(record, indent=1, sort_keys=True))
for item in problems:
    print("PROBLEM:", item, file=sys.stderr)
sys.exit(1 if problems else 0)
PY
echo "suites on torch:"
ls -la "$DATA/suites"
(( PROBLEMS == 0 )) || { echo "deploy checks failed: see the PROBLEM lines" >&2; exit 2; }
echo "deploy-complete $COMMIT"
