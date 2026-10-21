"""Flow-meter pair mimics as a suite: two biased flow meters at the ends of one candidate HIF line.

    python -m research.rl.build_mimic_suite --per-variant 24 --seed 20261015 --profile classifier_gated_diagnostics \\
        --output output/rl_hard_draw_20261015/mimic/development.json

The roots come from research.hypothesis_ranking.expert_e2e.build_mimic_roots
(the generator's own two-meter admission on clean corpus windows, same-sign
and opposite-sign variants).  They are ordinary multi_measurement roots to
the environment and the audit; the suite records each root's mimic variant
under ``audit.mimic`` so a report can read it as a stratum.  Both suspicion-
family profiles share the data contract, so the declared profile is the only
field rewritten.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-variant", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--profile", default="classifier_gated_diagnostics")
    parser.add_argument("--split", default="development", choices=("development", "dagger_train", "train"),
                        help="grouping.split stamped on every root; a training suite the DAgger collector rolls "
                             "out on needs dagger_train (its supervision policy refuses development roots)")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    from psse_env.evidence_profile import is_suspicion_profile, validate_evidence_profile
    from research.hypothesis_ranking.expert_e2e import build_mimic_roots

    profile = validate_evidence_profile(args.profile)
    if not is_suspicion_profile(profile):
        raise SystemExit("mimic roots carry the suspicion-family data contract; choose a suspicion-family profile")
    roots, families = build_mimic_roots(int(args.seed), int(args.per_variant))
    for root in roots:
        scenario_id = str(root["execution"]["scenario_id"])
        root["execution"]["evidence_profile"] = profile
        root.setdefault("audit", {})["mimic"] = {"variant": families[scenario_id]}
        root.setdefault("grouping", {})["split"] = str(args.split)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(roots, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"roots": len(roots), "variants": dict(Counter(families.values())), "profile": profile,
                      "split": str(args.split), "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
