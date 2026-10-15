"""Reuse a completed frozen-R2 probe for a *single* controlled update arm.

Checks code/config/model/root/seed identity, recomputes rewards from full
episode records, and starts a new manifest at update. Never runs a model,
modifies the source store, or reuses a later iteration's off-policy samples.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from psse_env.rl.reward import RewardWeights, episode_reward, load_expert_reference
from psse_env.rl.rollout import TrajectoryStore, sample_seed, scenario_root
from psse_env.rl.train import RLConfig, Manifest, iteration_dir, iteration_roots, load_suite, source_revision

ARM_FIELDS = {"output_dir", "reward", "advantage_method", "advantage_scale", "update", "update_checkpoint_rows",
              "development_suite", "evaluate_every_iteration", "evaluation_workers_per_policy"}


def fork(source: RLConfig, target: RLConfig) -> dict:
    source.validate()
    target.validate()
    if target.update_settings(0).optimizer_steps != 1 or target.policy_source != "adapter":
        raise ValueError("a frozen-rollout arm requires one optimizer step of an adapter")
    if Path(target.output_dir).exists():
        raise ValueError("target output already exists; refuse to mix experiments")
    source_dir = iteration_dir(source, 0)
    provenance = json.loads((source_dir / "rollout_config.json").read_text())
    if provenance["config"] != source.as_dict() or provenance["source_revision"] != source_revision():
        raise ValueError("source config or code revision changed after the frozen rollout")
    differences = [key for key in source.as_dict() if key not in ARM_FIELDS and
                   source.as_dict()[key] != target.as_dict()[key]]
    if differences:
        raise ValueError(f"sampling/environment differs: {differences}")
    summary = json.loads((source_dir / "rollout_summary.json").read_text())
    if summary.get("adapter") != source.initial_adapter:
        raise ValueError("source rollout did not use the frozen initial adapter")
    roots = {scenario_root(row) for row in iteration_roots(load_suite(source.training_suite), iteration=0,
                                                          roots_per_iteration=source.roots_per_iteration, seed=source.seed)}
    store = TrajectoryStore(source_dir / "trajectories")
    if set(store.roots()) != roots:
        raise ValueError("source probe is incomplete or has extra roots")
    reference = load_expert_reference(target.expert_reference)
    weights = RewardWeights.from_mapping(target.reward)
    if weights.tool_regret_cost and roots - set(reference):
        raise ValueError("expert reference is missing roots")
    # Validate and rescore everything before creating the target directory.
    groups = []
    for root, group in store.iter_groups():
        if len(group) != source.samples_per_root or {t.sample_index for t in group} != set(range(source.samples_per_root)):
            raise ValueError("source group is incomplete or duplicated")
        for trajectory in group:
            if (not trajectory.pairing_ok or trajectory.root != root or
                trajectory.episode_seed != sample_seed(source.seed, root, trajectory.sample_index) or
                Path(trajectory.policy_identity.get("adapter_path", "")) != Path(source.initial_adapter)):
                raise ValueError("source trajectory does not match frozen adapter, pairing, root, or seed")
            trajectory.reward = episode_reward(trajectory.evaluation, family=trajectory.family,
                                               expert_reference_calls=reference.get(root), weights=weights).as_dict()
        groups.append((root, group))
    output = iteration_dir(target, 0)
    target_store = TrajectoryStore(output / "trajectories")
    for root, group in groups:
        target_store.write_group(root, group)
    receipt = {"source": str(source_dir), "source_revision": provenance["source_revision"],
               "roots": sorted(roots), "source_config": source.as_dict(), "target_config": target.as_dict(),
               "reward_recomputed": True, "next_phase": "update", "optimizer_steps": 1}
    (output / "frozen_rollout_fork.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    Manifest(phase="update", current_adapter=target.initial_adapter).save(Path(target.output_dir) / "manifest.json")
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    args = parser.parse_args(argv)
    result = fork(RLConfig.load(args.source_config), RLConfig.load(args.target_config))
    print(json.dumps({"roots": len(result["roots"]), "next_phase": result["next_phase"], "reward_recomputed": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
