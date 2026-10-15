"""The RL iteration driver: sample, score, update, evaluate, checkpoint.

    python -m psse_env.rl.train --config rl.json --iterations 1          # run the next iterations
    python -m psse_env.rl.train --config rl.json --phase rollout         # one phase of the current iteration
    python -m psse_env.rl.train --config rl.json --phase update
    python -m psse_env.rl.train --config rl.json --status

Every phase is resumable: rollouts are stored per root group, the update
writes the next adapter atomically, and ``manifest.json`` records where the
run stands.  A phase is a separate process on the cluster
(research/hpc/rl_pilot_20261014/rl_pilot.sbatch) so a preemption loses at
most the root group in flight.

The config (JSON) names the training and development suites, the starting
adapter and base model, the evidence profile and environment options the
pipeline cell used, the sampling, reward, advantage and update settings.
See docs/rl_experiments_20261014.md for a worked example.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.rl.advantages import GroupAdvantages, group_advantages, summarize_groups
from psse_env.rl.reward import RewardWeights, load_expert_reference
from psse_env.rl.rollout import Trajectory, TrajectoryStore, rollout_roots, scenario_root
from psse_env.rl.rows import build_rows
from psse_env.rl.update import UpdateSettings, policy_gradient_update

TRAIN_CONTRACT = "rl_iteration_driver_v1"
PHASES = ("rollout", "update", "evaluate")


@dataclass
class RLConfig:
    output_dir: str
    training_suite: str
    development_suite: str | None
    initial_adapter: str
    base_model: str
    base_revision: str
    architecture: str | None = None
    prompt_profile: str | None = None
    evidence_profile: str = "classifier_gated_diagnostics"
    triage_classifier: str | None = None
    progress_basis: str | None = None
    normalized_residual_threshold: float | None = 4.0
    expert_variant: str = "baseline"
    hif_search_profile: str = "research"
    max_steps: int = 40
    samples_per_root: int = 4
    seed: int = 20261014
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int | None = None
    roots_per_iteration: int | None = None
    max_length: int = 32768
    advantage_method: str = "grpo"
    advantage_scale: str = "group"
    expert_reference: str | None = None
    load_in_4bit: bool = True
    local_files_only: bool = True
    evaluate_every_iteration: bool = True
    evaluation_workers_per_policy: int = 1
    #: ``adapter`` samples the LoRA policy; ``expert`` rolls the observable
    #: teacher out instead (a wiring check of the environment, the audit and
    #: the store on a CPU: its episodes carry no generation record, so the
    #: update phase has nothing to train on and carries the adapter over).
    policy_source: str = "adapter"
    reward: dict[str, Any] = field(default_factory=dict)
    update: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RLConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {name for name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.samples_per_root < 2:
            raise ValueError("samples_per_root must be at least 2 for a group-relative advantage")
        if self.max_steps <= 0 or self.max_length <= 0:
            raise ValueError("max_steps and max_length must be positive")
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if self.policy_source not in {"adapter", "expert"}:
            raise ValueError("policy_source must be adapter or expert")
        RewardWeights.from_mapping(self.reward)
        UpdateSettings(**self.update).validate()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    contract: str = TRAIN_CONTRACT
    iteration: int = 0
    phase: str = "rollout"
    current_adapter: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, *, initial_adapter: str) -> "Manifest":
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                contract=str(payload.get("contract") or TRAIN_CONTRACT),
                iteration=int(payload.get("iteration") or 0),
                phase=str(payload.get("phase") or "rollout"),
                current_adapter=str(payload.get("current_adapter") or initial_adapter),
                history=list(payload.get("history") or []),
            )
        return cls(current_adapter=initial_adapter)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("scenarios") or payload.get("rows") or payload.get("development") or []
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"suite {path} holds no scenarios")
    return [dict(row) for row in rows]


def iteration_roots(scenarios: Sequence[Mapping[str, Any]], *, iteration: int, roots_per_iteration: int | None, seed: int) -> list[dict[str, Any]]:
    """The roots of one iteration: all of them, or a seeded slice that cycles through the suite."""

    if roots_per_iteration is None or roots_per_iteration >= len(scenarios):
        return [dict(row) for row in scenarios]
    import random

    order = list(range(len(scenarios)))
    random.Random(seed).shuffle(order)
    start = (iteration * roots_per_iteration) % len(order)
    chosen = [order[(start + offset) % len(order)] for offset in range(roots_per_iteration)]
    return [dict(scenarios[index]) for index in chosen]


def configure_research_environment(config: RLConfig) -> Callable[..., Any]:
    """Point the research runner's environment options at the cell's profile and return its factory."""

    import scripts.run_dagger_research as research

    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = config.evidence_profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = config.normalized_residual_threshold
    research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = (
        str(Path(config.triage_classifier).resolve()) if config.triage_classifier else None
    )
    if config.progress_basis:
        research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = config.progress_basis
    research.RESEARCH_EXPERT_OPTIONS["variant"] = config.expert_variant
    return research.resolve_environment_factory(config.hif_search_profile, config.evidence_profile)


def sampling_policy_factory(config: RLConfig, adapter: str, env_factory: Callable[..., Any] | None = None) -> Callable[[int], Any]:
    if config.policy_source == "expert":
        import scripts.run_dagger_research as research

        if env_factory is None:
            raise ValueError("the expert policy source needs the environment factory")
        return lambda seed: research.research_expert_policy(env_factory)

    from psse_env.dagger.research_policy_factory import GenerationOptions, research_gemma_policy_factory

    def factory(seed: int) -> Any:
        return research_gemma_policy_factory(
            adapter,
            base_model=config.base_model,
            base_revision=config.base_revision,
            load_in_4bit=config.load_in_4bit,
            local_files_only=config.local_files_only,
            prompt_profile=config.prompt_profile,
            architecture=config.architecture,
            generation=GenerationOptions(
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=int(seed),
            ),
        )

    return factory


def iteration_dir(config: RLConfig, iteration: int) -> Path:
    return Path(config.output_dir) / f"iteration_{iteration:03d}"


def phase_rollout(config: RLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    from psse_env.dagger.release_factories import deterministic_case_loader

    out = iteration_dir(config, manifest.iteration)
    out.mkdir(parents=True, exist_ok=True)
    scenarios = iteration_roots(
        load_suite(config.training_suite),
        iteration=manifest.iteration,
        roots_per_iteration=config.roots_per_iteration,
        seed=config.seed,
    )
    (out / "roots.json").write_text(json.dumps([scenario_root(s) for s in scenarios], indent=1) + "\n", encoding="utf-8")
    env_factory = configure_research_environment(config)
    store = TrajectoryStore(out / "trajectories")
    weights = RewardWeights.from_mapping(config.reward)
    reference = load_expert_reference(config.expert_reference)
    summary = rollout_roots(
        scenarios,
        store=store,
        samples=config.samples_per_root,
        base_seed=config.seed + manifest.iteration,
        env_factory=env_factory,
        policy_factory=sampling_policy_factory(config, manifest.current_adapter, env_factory),
        max_steps=config.max_steps,
        case_loader=deterministic_case_loader,
        weights=weights,
        expert_reference=reference or None,
        progress=progress,
    )
    summary.update(adapter=manifest.current_adapter, iteration=manifest.iteration, roots=len(scenarios))
    _write_json(out / "rollout_summary.json", summary)
    return summary


def score_groups(config: RLConfig, store: TrajectoryStore) -> tuple[list[tuple[list[Trajectory], GroupAdvantages]], dict[str, Any]]:
    groups: list[tuple[list[Trajectory], GroupAdvantages]] = []
    by_root: dict[str, GroupAdvantages] = {}
    for root, trajectories in store.iter_groups():
        advantages = group_advantages(
            [trajectory.total_reward for trajectory in trajectories],
            method=config.advantage_method,
            scale=config.advantage_scale,
            root=root,
        )
        groups.append((trajectories, advantages))
        by_root[root] = advantages
    summary = summarize_groups(by_root)
    successes = sum(
        1 for trajectories, _ in groups for trajectory in trajectories if trajectory.evaluation.get("truth_audited_task_success")
    )
    episodes = sum(len(trajectories) for trajectories, _ in groups)
    summary.update(
        episodes=episodes,
        success_rate=(successes / episodes) if episodes else None,
        mean_calls=(sum(int(t.reward.get("calls") or 0) for ts, _ in groups for t in ts) / episodes) if episodes else None,
        unpaired_trajectories=sum(1 for ts, _ in groups for t in ts if not t.pairing_ok),
    )
    return groups, summary


def _load_trainable_model(config: RLConfig, adapter: str) -> tuple[Any, Any, tuple[str, ...]]:
    """The base in 4-bit with the adapter attached trainable, plus its processor and side inputs."""

    from psse_env.sft.research_cli import ResearchTrainerSettings, _load_research_base, attach_research_adapter
    from psse_env.sft.training import LoraSettings, infer_required_side_input_names

    settings = ResearchTrainerSettings(
        model_name=config.base_model,
        revision=config.base_revision,
        output_dir=str(Path(config.output_dir) / "_trainer_scratch"),
        architecture=config.architecture,
        max_length=config.max_length,
        load_in_4bit=config.load_in_4bit,
        local_files_only=config.local_files_only,
        initial_adapter_path=adapter,
    )
    settings.validate()
    model = _load_research_base(settings)
    model, _attachment = attach_research_adapter(model, settings=settings, lora=LoraSettings())
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.base_model, revision=config.base_revision, local_files_only=config.local_files_only
    )
    side_inputs = infer_required_side_input_names(model, processor, config.base_model)
    return model, processor, side_inputs


def _prompt_functions(config: RLConfig, adapter: str, processor: Any) -> tuple[Callable[[Mapping[str, Any]], str], Callable[[str], tuple[list[int], int]]]:
    """The policy's own prompt renderer and tokenizer, bound to the training processor."""

    from psse_env.dagger.release_factories import _ModelBundle
    from psse_env.dagger.research_policy_factory import _CanonicalResearchNativePolicy

    bundle = _ModelBundle(model=None, processor=processor, model_id=config.base_model,
                          model_revision=config.base_revision, adapter_snapshot_path=str(adapter))
    policy = _CanonicalResearchNativePolicy(bundle, architecture=str(config.architecture or "gemma4_unified"))

    def render(observation: Mapping[str, Any]) -> str:
        rendered, _tools, _parameters = policy.render_prompt(observation)
        return rendered

    return render, policy.prompt_token_ids


def phase_update(config: RLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    out = iteration_dir(config, manifest.iteration)
    store = TrajectoryStore(out / "trajectories")
    groups, score_summary = score_groups(config, store)
    _write_json(out / "advantages.json", {
        **score_summary,
        "groups": {adv.root: {"rewards": list(adv.rewards), "advantages": list(adv.advantages), "informative": adv.informative}
                   for _, adv in groups},
    })
    next_adapter = out / "adapter"
    report: dict[str, Any] = {"iteration": manifest.iteration, "adapter_in": manifest.current_adapter, "scores": score_summary}
    trainable = any(adv.informative for _, adv in groups) and any(
        trajectory.pairing_ok for trajectories, adv in groups if adv.informative for trajectory in trajectories
    )
    if not trainable:
        report["update"] = None
        report["note"] = "no informative, paired group: the adapter is carried over unchanged"
        if next_adapter.exists():
            shutil.rmtree(next_adapter)
        shutil.copytree(manifest.current_adapter, next_adapter)
        _write_json(out / "update_report.json", report)
        return report
    model, processor, side_inputs = _load_trainable_model(config, manifest.current_adapter)
    try:
        render, prompt_ids = _prompt_functions(config, manifest.current_adapter, processor)
        rows, rows_report = build_rows(
            groups, render_prompt=render, prompt_token_ids=prompt_ids,
            max_length=config.max_length, required_side_inputs=side_inputs,
        )
        report["rows"] = rows_report
        if not rows:
            report["update"] = None
            report["note"] = "no training rows after pairing and rendering checks"
            if next_adapter.exists():
                shutil.rmtree(next_adapter)
            shutil.copytree(manifest.current_adapter, next_adapter)
        else:
            settings = UpdateSettings(**{**config.update, "seed": config.seed + manifest.iteration})
            update = policy_gradient_update(model, rows, settings, progress=progress)
            report["update"] = update.as_dict()
            temporary = out / "adapter.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            model.save_pretrained(str(temporary))
            if next_adapter.exists():
                shutil.rmtree(next_adapter)
            temporary.rename(next_adapter)
    finally:
        del model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch optional in tests
            pass
    report["adapter_out"] = str(next_adapter)
    _write_json(out / "update_report.json", report)
    return report


def phase_evaluate(config: RLConfig, manifest: Manifest) -> dict[str, Any] | None:
    """Greedy evaluation of the iteration's new adapter on the development suite."""

    if not config.development_suite or not config.evaluate_every_iteration:
        return None
    from psse_env.dagger.release_factories import deterministic_case_loader
    from psse_env.dagger.evaluator import evaluate_rollout_suites
    from psse_env.dagger.research_policy_factory import clear_research_policy_cache, research_gemma_policy_factory
    from scripts.run_dagger_research import PAIRED_EVALUATION_CONTRACT

    out = iteration_dir(config, manifest.iteration)
    adapter = str(out / "adapter")
    env_factory = configure_research_environment(config)
    scenarios = load_suite(config.development_suite)
    policy = research_gemma_policy_factory(
        adapter, base_model=config.base_model, base_revision=config.base_revision,
        load_in_4bit=config.load_in_4bit, local_files_only=config.local_files_only,
        prompt_profile=config.prompt_profile, architecture=config.architecture,
    )
    result = evaluate_rollout_suites(
        {"standard_success": scenarios}, env_factory=env_factory,
        policy_factory=lambda **_kwargs: policy, max_steps=config.max_steps, seed=config.seed,
        case_loader=deterministic_case_loader, **PAIRED_EVALUATION_CONTRACT,
    ).as_dict()
    overall = result["suite_metrics"]["overall"]
    episodes = result["suite_metrics"]["episodes"]
    summary = {
        "iteration": manifest.iteration,
        "adapter": adapter,
        "roots": len(scenarios),
        "truth_audited_task_success": sum(1 for e in episodes if e.get("truth_audited_task_success")),
        "mean_steps": overall.get("mean_steps"),
        "false_commit_count": overall.get("false_commit_count"),
        "loop_episodes": overall.get("loop_episodes"),
        "invalid_action_count": overall.get("invalid_action_count"),
    }
    _write_json(out / "evaluation.json", {"summary": summary, "overall": overall})
    with (out / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            compact = {key: episode.get(key) for key in ("scenario_id", "family", "truth_audited_task_success", "steps", "false_commit_count", "loop_detected")}
            stream.write(json.dumps(compact, sort_keys=True) + "\n")
    del policy
    clear_research_policy_cache()
    return summary


def run_phase(config: RLConfig, phase: str, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any] | None:
    manifest_path = Path(config.output_dir) / "manifest.json"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = Manifest.load(manifest_path, initial_adapter=config.initial_adapter)
    if phase != manifest.phase:
        raise RuntimeError(f"the run is at phase {manifest.phase!r} of iteration {manifest.iteration}, not {phase!r}")
    started = time.perf_counter()
    if phase == "rollout":
        result = phase_rollout(config, manifest, progress=progress)
        manifest.phase = "update"
    elif phase == "update":
        result = phase_update(config, manifest, progress=progress)
        manifest.phase = "evaluate"
    elif phase == "evaluate":
        result = phase_evaluate(config, manifest)
        manifest.history.append({
            "iteration": manifest.iteration,
            "adapter": str(iteration_dir(config, manifest.iteration) / "adapter"),
            "evaluation": result,
        })
        manifest.current_adapter = str(iteration_dir(config, manifest.iteration) / "adapter")
        manifest.iteration += 1
        manifest.phase = "rollout"
    else:
        raise ValueError(f"unknown phase {phase!r}")
    manifest.save(manifest_path)
    print(json.dumps({"phase": phase, "iteration": manifest.iteration, "seconds": round(time.perf_counter() - started, 1),
                      "result": result}, default=str), flush=True)
    return result


def run_iterations(config: RLConfig, iterations: int, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> None:
    for _ in range(int(iterations)):
        manifest = Manifest.load(Path(config.output_dir) / "manifest.json", initial_adapter=config.initial_adapter)
        start_phase = PHASES.index(manifest.phase)
        for phase in PHASES[start_phase:]:
            run_phase(config, phase, progress=progress)


def status(config: RLConfig) -> dict[str, Any]:
    manifest = Manifest.load(Path(config.output_dir) / "manifest.json", initial_adapter=config.initial_adapter)
    return asdict(manifest)


def _print_progress(record: Mapping[str, Any]) -> None:
    print(json.dumps(dict(record), default=str), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=PHASES, default=None, help="run one phase of the current iteration")
    parser.add_argument("--iterations", type=int, default=1, help="iterations to run when no phase is given")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    config = RLConfig.load(args.config)
    if args.status:
        print(json.dumps(status(config), indent=2))
        return 0
    if args.phase:
        run_phase(config, args.phase, progress=_print_progress)
        return 0
    run_iterations(config, args.iterations, progress=_print_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
