#!/usr/bin/env bash
# Deploy and submit the comparison-teacher cell on the hard draw (2026-10-21) from Windows.
#
#   bash research/hpc/full_pipeline_20260907/launch_compare_cell_20261021.sh [commit40]
#
# One DAgger round of the frozen screen student (the 2026-10-01 ranked cell's
# round-2 adapter) on the hard training draw under the ledger_ranked_compare
# teacher (overrides/compare_hard_20261021.env), chain d0 -> r1c -> r1t -> r1e.
# The suites are the hard draw's screen files, copied from the RL pilot's data
# directory into out/suite with a copied-suite receipt before submission; stage 0
# then regenerates only the expert aggregate (and links the frozen adapter).
# Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL=$REPO/research/hpc/full_pipeline_20260907
# git.exe takes the POSIX form of the repository path literally once MSYS path conversion is off: work from inside it.
cd "$REPO"
BRANCH=${BRANCH:-codex/classifier-triage-20261004}
COMMIT=${1:-$(git rev-parse "$BRANCH")}
BASE=${BASE:-d2206f3}                        # the RL pilot's source clone HEAD (2026-10-19)
RL=/scratch/yx3882/rl_pilot_20261014
HARD=$RL/data/hard_draw_20261015
P=/scratch/yx3882/research_full_pipeline_20261021_compare
SUBMIT=${SUBMIT:-1}
export MSYS_NO_PATHCONV=1

[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts)" ]] \
  || { echo "commit the pipeline sources first (tracked modifications under research/, psse_env/ or scripts/)" >&2; exit 2; }

echo "== cell: $P from $COMMIT"
PIPE=$P SOURCE_CELL=$RL BASE=$BASE BRANCH=$BRANCH OVERRIDES=compare_hard_20261021.env SUBMIT=0 \
REMOTE_PREPARE="set -e
mkdir -p $P/out/suite
cp $HARD/hard_training_screen.json $P/out/suite/r1_training.json
cp $HARD/hard_development_screen.json $P/out/suite/development.json
python3 - <<'PY'
import hashlib, json
from collections import Counter
from pathlib import Path
suite = Path('$P/out/suite')
hard = Path('$HARD')
def stratum(root):
    audit = root.get('audit') or {}
    mimic = audit.get('mimic')
    if isinstance(mimic, dict) and mimic.get('variant'):
        return 'mimic:' + str(mimic['variant']).removeprefix('mimic_')
    family = str(root['grouping']['scenario_family'])
    if family in ('parameter', 'measurement+parameter'):
        return family + ':' + str((audit.get('parameter_ranking') or {}).get('stratum') or 'unknown')
    return family
files = {}
for name, source in (('r1_training.json', 'hard_training_screen.json'), ('development.json', 'hard_development_screen.json')):
    rows = json.loads((suite / name).read_text(encoding='utf-8'))
    assert isinstance(rows, list) and rows, name
    files[name] = {
        'copied_from': str(hard / source),
        'sha256': hashlib.sha256((suite / name).read_bytes()).hexdigest(),
        'roots': len(rows),
        'families': dict(sorted(Counter(r['grouping']['scenario_family'] for r in rows).items())),
        'strata': dict(sorted(Counter(stratum(r) for r in rows).items())),
        'evidence_profiles': dict(sorted(Counter(str((r.get('execution') or {}).get('evidence_profile')) for r in rows).items())),
    }
receipt = {
    'contract': 'research_full_pipeline_suite_copied_v1',
    'evidence_profile': 'suspicion_gated_diagnostics',
    'expert_variant': 'ledger_ranked_compare',
    'expert_variant_note': 'the receipt names the teacher of this cell; the suite is the hard draw of 2026-10-15 (build_suite physical admission, rank allowance 99, effect band 1.5-3, stratum plans; docs/rl_experiments_20261014.md section 4), re-executed under suspicion_gated_diagnostics, not a draw by this teacher',
    'rounds': 1,
    'files': files,
}
(suite / 'manifest.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + chr(10), encoding='utf-8')
(suite.parent / 'suite.done').write_text(json.dumps(receipt, indent=2, sort_keys=True) + chr(10), encoding='utf-8')
print(json.dumps({name: (f['roots'], f['strata']) for name, f in files.items()}))
PY
echo prepared" \
  bash "$CELL/deploy_cell_from_windows.sh" "$COMMIT"

if [[ "$SUBMIT" == 1 ]]; then
  wsl -- ssh torch "cd $P && CHAIN='d0 r1c r1t r1e' bash submit_pipeline.sh"
  echo "cell submitted"
else
  echo "deployed; not submitted (SUBMIT=0): wsl -- ssh torch \"cd $P && CHAIN='d0 r1c r1t r1e' bash submit_pipeline.sh\""
fi
echo "status: MSYS_NO_PATHCONV=1 wsl -- ssh torch bash $P/status_pipeline.sh"
echo "after stage 0: wsl -- ssh torch \"cd $P && \$(sed -n 's/^PY=//p' pipeline.env) source/research/hpc/full_pipeline_20260907/check_suite_exclusion.py --d0-raw out/d0/aggregate.raw.jsonl --suite-dir out/suite --reference-d0-raw /scratch/yx3882/research_full_pipeline_20261001_ranked/out/d0/aggregate.raw.jsonl\""
