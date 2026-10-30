#!/usr/bin/env bash
#SBATCH --job-name=fp-r2wgate
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/r2wgate-%j.out
# Round 2 from R1 (2026-10-30, note section 10): run after r2t, before r2e and
# the eight evaluations.  Exit 0 only if the training receipt says the pass
# started from R1's adapter (ROUND_INIT=previous) on the cumulative mixture the
# BC0-start pass trained on (same rows, same sha256); otherwise exit 3, so
# nothing evaluates an adapter trained some other way.
set -euo pipefail
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
PY=/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python
echo "[r2wgate] job ${SLURM_JOB_ID:-manual} node ${SLURMD_NODENAME:-?} $(date -u +%FT%TZ)"
"$PY" - "$P/out" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
def read(path):
    return json.loads((out / path).read_text(encoding="utf-8"))
done = read("r2/training.done")
r1 = read("r1/training.done")
bc0init = read("r2_bc0init_20261030/training.done")
collection = read("r2/collection.done")
mixture = hashlib.sha256((out / "r2" / "collection" / "round1.train.jsonl").read_bytes()).hexdigest()
checks = {
    "round_init == previous": done.get("round_init") == "previous",
    "round_aggregate == cumulative": done.get("round_aggregate") == "cumulative",
    "initialization is R1's adapter": done.get("init_adapter_path") == str(out / "r1" / "training" / "lora"),
    "initialization tree == R1's candidate tree": done.get("init_adapter_tree_sha256") == r1.get("candidate_tree_sha256"),
    "collecting student tree == R1's candidate tree": done.get("student_tree_sha256") == r1.get("candidate_tree_sha256"),
    "mixture rows == 1738": done.get("mixture_rows") == 1738 == collection.get("mixture_rows"),
    "optimizer steps == 435": done.get("max_steps") == 435,
    "mixture sha256 == the BC0-start pass's": mixture == collection.get("mixture_sha256")
        == read("r2_bc0init_20261030/collection.done").get("mixture_sha256"),
    "research run passed": done.get("research_run_passed") is True,
    "candidate differs from the BC0-start R2": done.get("candidate_tree_sha256") != bc0init.get("candidate_tree_sha256"),
}
print(json.dumps({"checks": checks, "passed": all(checks.values()), "receipt": done}, indent=1, sort_keys=True))
sys.exit(0 if all(checks.values()) else 3)
PY
echo "fp-r2wgate-passed $(date -u +%FT%TZ)"
