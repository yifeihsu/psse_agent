#!/usr/bin/env bash
# Copy the LLM leg's results from the cluster work directory into the Windows checkout.
#
#   bash research/hpc/classifier_triage_20261004/fetch_results.sh [run ...]
#
# A run is a directory under the work directory's out/ (prompt_top5,
# prompt_top10_signed_e3, ...); the default is the two one-pass runs.  Two ssh
# calls per run: one lists the result files the cluster has with their
# checksums, one streams them as a compressed tar through ssh stdout (scp from
# WSL cannot write to the Windows side).  The copies are verified against the
# listed checksums, so a file still being written is refused, not kept.
set -euo pipefail
RUNS=("$@")
[[ ${#RUNS[@]} -gt 0 ]] || RUNS=(prompt_top5 prompt_top10_signed)
WORK=${WORK:-/scratch/yx3882/classifier_triage_20261004}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
DEST=${DEST:-$REPO/output/classifier_triage_20261004/llm_results}
FILES="scores.json probabilities.json done.json research_run.json trainer_state.json"
export MSYS_NO_PATHCONV=1

for run in "${RUNS[@]}"; do
  listing=$(wsl -- ssh -o NumberOfPasswordPrompts=0 torch bash -s <<EOF
cd $WORK/out/$run 2>/dev/null || exit 0
for name in $FILES; do
  if [ -s "\$name" ]; then sha256sum "\$name"; fi
done
EOF
)
  if [[ -z "$listing" ]]; then
    echo "$run: no result file on the cluster"
    continue
  fi
  names=$(printf '%s\n' "$listing" | awk '{print $2}' | tr '\n' ' ')
  part=$DEST/$run.part
  rm -rf "$part"
  mkdir -p "$part" "$DEST/$run"
  wsl -- ssh torch "tar czf - -C $WORK/out/$run $names" | tar xzf - -C "$part"
  (cd "$part" && printf '%s\n' "$listing" | sha256sum -c --quiet -) \
    || { echo "$run: the download does not match the cluster files" >&2; exit 2; }
  for name in $names; do
    mv "$part/$name" "$DEST/$run/$name"
    echo "$run/$name: $(stat -c %s "$DEST/$run/$name") bytes"
  done
  rmdir "$part"
done
