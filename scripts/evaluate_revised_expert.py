"""Resume a revised-code expert or adapter (BC0, R1, R2) evaluation on the original frozen160.

Canonical evaluator trajectories, truth isolation, seed derivation, audits and
aggregation are unchanged. Only its per-episode invocation is checkpointed.
No scenarios are regenerated and no sensor noise is redrawn.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from functools import partial
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from psse_env.dagger import evaluator as canonical
from psse_env.dagger.release_factories import deterministic_case_loader, ObservableExpertPolicy
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.providers.hif_continuation import model_fingerprint
from psse_env.research_models import get_research_model_spec
from scripts.run_dagger_research import research_diagnostic_environment_factory
from psse_env.evidence_profile import DEFAULT_EVIDENCE_PROFILE, EVIDENCE_PROFILES

FROZEN_SHA256 = "3a088ebc9dd7b847511d70ecd2fabd945979c66f0303cd5f964001fc958f5262"
EVALUATOR_SEED = 20260912
SUITE_NAME = "standard_success"
EXPECTED_ROOTS = 160
CONTRACT = "frozen160_revised_code_evaluation_v1"
#: Frozen LoRA students of the 2026-09-21 pipeline; each needs --adapter-path.
ADAPTER_POLICIES = ("bc0", "r1", "r2")
#: Truth-audited successes each policy scored in the original pipeline evaluation,
#: used only when no per-root baseline report is supplied.
PREVIOUS_REPORTED_SUCCESSES = {"expert": 142, "bc0": 132, "r1": 140, "r2": 141}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def source_identity():
    files = {Path(__file__).resolve(), REPO/"scripts/run_dagger_research.py", REPO/"mcp_server/case14.m"}
    files.update(path for path in REPO.glob("*.py") if not path.name.startswith("test_"))
    for folder in ("psse_env", "mcp_server", "tools", "three_phase_nlm", "IEEE_14_OpenDSS",
                   "three_phase_model", "Transmission", "logical_topology"):
        files.update(path for path in (REPO/folder).rglob("*.py")
                     if not path.name.startswith("test_") and "tests" not in path.parts)
    hashes = {path.relative_to(REPO).as_posix(): file_sha256(path) for path in sorted(files) if path.is_file()}
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None  # Exported source snapshots need not carry .git.
    return {"git_head": head, "runtime_source_sha256": hashes,
            "runtime_source_set_sha256": digest(hashes), "physical_model_sha256": model_fingerprint(None)}


def adapter_identity(path):
    directory = Path(path).expanduser().resolve(strict=True)
    if not directory.is_dir() or not (directory/"adapter_config.json").is_file():
        raise ValueError("Adapter requires a directory with adapter_config.json")
    selected = sorted(p for p in directory.rglob("*") if p.is_file() and
        (p.name == "adapter_config.json" or p.name.startswith("adapter_model")))
    if not any(p.name.startswith("adapter_model") for p in selected):
        raise ValueError("Adapter weights are missing")
    hashes = {p.relative_to(directory).as_posix(): file_sha256(p) for p in selected}
    return {"adapter_path": str(directory), "adapter_files_sha256": hashes,
            "adapter_content_sha256": digest(hashes)}


def load_frozen_scenarios(path, *, expected_sha256=FROZEN_SHA256, expected_roots=EXPECTED_ROOTS):
    path = Path(path)
    if file_sha256(path) != expected_sha256:
        raise ValueError("Frozen scenario bytes differ from the requested original development suite")
    rows = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or len(rows) != expected_roots:
        raise ValueError(f"Expected exactly {expected_roots} frozen scenario envelopes")
    suites = {SUITE_NAME: rows}
    canonical.validate_release_scenario_suites(suites)
    manifest = canonical.fingerprint_evaluation_suites(suites, seed=EVALUATOR_SEED,
        required_suites=[SUITE_NAME], minimum_roots_per_suite=expected_roots)
    entries = manifest["episode_manifest"]
    if (len({r["scenario_id"] for r in entries}) != expected_roots
        or len({r["physical_root"] for r in entries}) != expected_roots
        or any(r["scenario_index"] != 0 for r in entries)):
        raise ValueError("Frozen evaluation requires distinct scenario IDs and physical roots")
    return suites, manifest


class LazyPolicyFactory:
    """Load one actor just as evaluate_paired_adapters does, only if needed."""
    def __init__(self, builder):
        self.builder, self.policy = builder, None

    def __call__(self, **_kwargs):
        if self.policy is None:
            self.policy = self.builder()
        return self.policy


def policy_builder(policy, options, *, model_loader=None):
    if policy == "expert":
        return lambda: ObservableExpertPolicy(ExpertPolicyOracle(
            process_oracle=research_diagnostic_environment_factory(
                evidence_profile=options.get("evidence_profile", DEFAULT_EVIDENCE_PROFILE)).process_oracle))
    if policy not in ADAPTER_POLICIES:
        raise ValueError("policy must be expert, bc0, r1 or r2")
    if model_loader is None:
        from psse_env.dagger.research_policy_factory import research_gemma_policy_factory
        model_loader = research_gemma_policy_factory
    return lambda: model_loader(options["adapter_path"], base_model=options["base_model"],
        base_revision=options["base_revision"], load_in_4bit=options["load_in_4bit"],
        local_files_only=True, trust_remote_code=options["trust_remote_code"],
        prompt_profile=options["prompt_profile"], architecture=options["architecture"])


class EpisodeCheckpoints:
    def __init__(self, directory, run_binding, episode_manifest, *, progress=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_binding = run_binding
        self.expected = {row["episode_key"]: row for row in episode_manifest}
        if len(self.expected) != len(episode_manifest):
            raise ValueError("Duplicate expected episode keys")
        self.completed = {}
        self.progress = progress

    def execute(self, original, evaluator, *, suite, scenario, scenario_index, episode_seed):
        sid = canonical._scenario_id(scenario, scenario_index)
        key = f"{suite}:{sid}:{scenario_index}"
        expected = self.expected.get(key)
        if expected is None or expected["seed"] != episode_seed:
            raise ValueError("Canonical episode identity/seed differs from the full frozen manifest")
        binding = {"run_binding_sha256": self.run_binding, "episode_key": key,
                   "episode_seed": episode_seed, "scenario_sha256": digest(scenario)}
        path = self.directory/(hashlib.sha256(key.encode()).hexdigest()+".json")
        origin = "resumed"
        if path.is_file():
            saved = json.loads(path.read_text())
            if saved.get("binding") != binding or saved.get("episode_sha256") != digest(saved.get("episode")):
                raise ValueError(f"Incomplete or inconsistent episode checkpoint: {path}")
            episode = canonical.EpisodeEvaluation(**saved["episode"])
        else:
            origin = "executed"
            episode = original(evaluator, suite=suite, scenario=scenario, scenario_index=scenario_index,
                               episode_seed=episode_seed)
        if (episode.episode_key != key or episode.seed != episode_seed
            or episode.physical_root != expected["physical_root"]):
            raise ValueError("Episode result is not bound to the expected frozen root")
        if origin == "executed":
            payload = episode.as_dict()
            atomic_json(path, {"binding": binding, "episode": payload, "episode_sha256": digest(payload)})
        self.completed[key] = origin
        if self.progress:
            self.progress({"event": "episode_checkpoint", "episode_key": key, "origin": origin,
                "completed": len(self.completed), "expected": len(self.expected),
                "task_success": episode.truth_audited_task_success,
                "terminal_outcome": episode.terminal_outcome, "steps": episode.steps})
        return episode

    @contextmanager
    def installed(self):
        # Scope the process-local wrapper to this evaluation. It neither edits
        # evaluator source nor changes its auditing/seed/aggregation behavior.
        original = canonical.ClosedLoopRolloutEvaluator._run_episode

        def checkpointed(evaluator, **kwargs):
            return self.execute(original, evaluator, **kwargs)

        with patch.object(canonical.ClosedLoopRolloutEvaluator, "_run_episode", checkpointed):
            yield


def evaluate_checkpointed(suites, manifest, *, output_dir, run_binding, env_factory, policy_factory, progress=None):
    store = EpisodeCheckpoints(Path(output_dir)/"episodes", run_binding, manifest["episode_manifest"], progress=progress)
    with store.installed():
        result = canonical.evaluate_rollout_suites(suites, env_factory=env_factory, policy_factory=policy_factory,
            max_steps=40, seed=EVALUATOR_SEED, required_suites=[SUITE_NAME], minimum_suites=1,
            minimum_episodes_per_suite=1, minimum_roots_per_suite=1,
            require_release_environment=False, require_policy_identity=False,
            case_loader=deterministic_case_loader)
    if set(store.completed) != set(store.expected):
        raise ValueError("Missing completed root checkpoints; final report must not be certified")
    payload = result.as_dict()
    actual = payload["suite_metrics"]["episodes"]
    if (len(actual) != len(store.expected) or {e["episode_key"] for e in actual} != set(store.expected)
        or payload["suite_metrics"]["configuration"]["episode_manifest_sha256"] != manifest["episode_manifest_sha256"]):
        raise ValueError("Canonical final report does not cover the full frozen manifest")
    return payload, {"executed": sum(v == "executed" for v in store.completed.values()),
                     "resumed": sum(v == "resumed" for v in store.completed.values())}


def compare_baseline(payload, baseline_path, policy):
    current = payload["suite_metrics"]["overall"]
    if baseline_path is None:
        count = PREVIOUS_REPORTED_SUCCESSES[policy]
        return {"basis": "previous_reported_count; no per-root baseline supplied",
            "baseline_task_successes": count, "revised_task_successes": current["truth_audited_task_success_episodes"],
            "task_success_delta": current["truth_audited_task_success_episodes"]-count}
    baseline = json.loads(Path(baseline_path).read_text())
    previous = baseline["suite_metrics"]["overall"]
    old = {e["episode_key"]: e for e in baseline["suite_metrics"]["episodes"]}
    new = {e["episode_key"]: e for e in payload["suite_metrics"]["episodes"]}
    if set(old) != set(new) or any(old[k]["physical_root"] != new[k]["physical_root"] or old[k]["seed"] != new[k]["seed"] for k in old):
        raise ValueError("Baseline roots/seeds do not match the frozen revised evaluation")
    changes = [{"episode_key": key, "scenario_id": new[key]["scenario_id"], "family": new[key]["family"],
        "before": old[key]["truth_audited_task_success"], "after": new[key]["truth_audited_task_success"],
        "before_terminal": old[key]["terminal_outcome"], "after_terminal": new[key]["terminal_outcome"]}
        for key in sorted(new) if old[key]["truth_audited_task_success"] != new[key]["truth_audited_task_success"]]
    return {"basis": "matched_full_baseline_report", "baseline_sha256": file_sha256(baseline_path),
        "baseline_overall": previous, "revised_overall": current, "changed_task_outcomes": changes,
        "task_success_delta": current["truth_audited_task_success_episodes"]-previous["truth_audited_task_success_episodes"]}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", choices=("expert", *ADAPTER_POLICIES), required=True)
    p.add_argument("--scenarios", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--adapter-path", type=Path)
    p.add_argument("--baseline-eval", type=Path)
    p.add_argument("--model-choice", default="12b")
    p.add_argument("--base-model")
    p.add_argument("--base-revision")
    p.add_argument("--prompt-profile")
    p.add_argument("--architecture")
    p.add_argument("--no-load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--evidence-profile", choices=EVIDENCE_PROFILES, default=DEFAULT_EVIDENCE_PROFILE,
        help=("Default wls_gated_diagnostics: balanced SCADA/WLS detection, auxiliary streams only after a "
              "current WLS alarm; scada_only refuses them; historical sensor-assisted reproduction "
              "requires auxiliary_diagnostics"))
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if (args.policy in ADAPTER_POLICIES) != (args.adapter_path is not None):
        raise ValueError("--adapter-path is required only for the adapter policies bc0, r1 and r2")
    suites, manifest = load_frozen_scenarios(args.scenarios)
    identity = source_identity()
    environment_factory = partial(research_diagnostic_environment_factory, evidence_profile=args.evidence_profile)
    env = environment_factory()
    provider = env.wls_runner.__self__
    environment = {"chi_square_alpha": provider.chi2_alpha,
        "normalized_residual_threshold": provider.normalized_residual_threshold,
        "hif_alpha_grid_size": provider.hif_alpha_grid_size, "hif_r_grid_size": provider.hif_r_grid_size,
        "hif_max_scans": provider.hif_max_scans, "max_steps": env.max_steps,
        "action_budget_scope": "all_episode_actions", "evidence_profile": args.evidence_profile}
    if environment != {"chi_square_alpha": .01, "normalized_residual_threshold": 4.,
        "hif_alpha_grid_size": 7, "hif_r_grid_size": 9, "hif_max_scans": 10,
        "max_steps": 40, "action_budget_scope": "all_episode_actions", "evidence_profile": args.evidence_profile}:
        raise ValueError(f"Research environment differs from the requested original configuration: {environment}")
    del env, provider
    spec = get_research_model_spec(args.model_choice)
    from psse_env.dagger.preliminary_e2b_eval import MAX_INPUT_TOKENS, MAX_NEW_TOKENS
    if args.policy in ADAPTER_POLICIES and (MAX_INPUT_TOKENS, MAX_NEW_TOKENS) != (32768, 256):
        raise ValueError("Matched adapter evaluation requires RESEARCH_MAX_INPUT_TOKENS=32768 and "
                         "RESEARCH_MAX_NEW_TOKENS=256 before Python starts; module defaults differ")
    options = {"base_model": args.base_model or spec.model_id,
        "evidence_profile": args.evidence_profile,
        "base_revision": args.base_revision or spec.revision,
        "prompt_profile": args.prompt_profile or spec.prompt_profile,
        "architecture": args.architecture or spec.architecture,
        "load_in_4bit": not args.no_load_in_4bit, "local_files_only": True,
        "trust_remote_code": args.trust_remote_code,
        "generation_budget": {"max_input_tokens": MAX_INPUT_TOKENS, "max_new_tokens": MAX_NEW_TOKENS,
                              "do_sample": False, "temperature": 0.0}}
    adapter = adapter_identity(args.adapter_path) if args.adapter_path else None
    if adapter:
        options["adapter_path"] = adapter["adapter_path"]
    semantic = {"contract": CONTRACT, "policy": args.policy, "scenario_sha256": FROZEN_SHA256,
        "manifest_sha256": manifest["episode_manifest_sha256"], "evaluator_seed": EVALUATOR_SEED,
        "environment": environment, "source_set_sha256": identity["runtime_source_set_sha256"],
        "physical_model_sha256": identity["physical_model_sha256"],
        "policy_options": {k:v for k,v in options.items() if k != "adapter_path"},
        "adapter_content_sha256": adapter["adapter_content_sha256"] if adapter else None}
    binding = digest(semantic)
    output = args.output_dir.resolve()
    config_path = output/"run_configuration.json"
    if output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume for the identical run or a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    config = {**semantic, "run_binding_sha256": binding, "source_identity": identity,
        "adapter_identity": adapter, "scenario_path": str(args.scenarios.resolve()),
        "policy_factory": "ObservableExpertPolicy" if args.policy == "expert" else "research_gemma_policy_factory",
        "no_noise_redraw": True, "no_custom_physical_audit_or_truth_policy": True}
    if config_path.exists():
        prior = json.loads(config_path.read_text())
        if prior.get("run_binding_sha256") != binding:
            raise ValueError("Resume would mix different source/model/scenario/policy settings; use a new output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Nonempty output has no checkpoint configuration")
        atomic_json(config_path, config)
    filename = f"{args.policy}_eval.json"
    completed_path = output/"completed.json"
    if completed_path.exists():
        done = json.loads(completed_path.read_text())
        if done.get("run_binding_sha256") != binding or done.get("evaluation_sha256") != file_sha256(output/filename):
            raise ValueError("Completed marker disagrees with its evaluation")
        print(json.dumps({"already_complete": True, "output": str(output/filename)}), flush=True)
        return 0
    progress_path = output/"progress.jsonl"

    def progress(event):
        entry = {"time_utc": datetime.now(timezone.utc).isoformat(), **event}
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, sort_keys=True, allow_nan=False)+"\n")
            handle.flush()
        print(json.dumps(entry, sort_keys=True), flush=True)

    started = time.perf_counter()
    actor = LazyPolicyFactory(policy_builder(args.policy, options))
    payload, counts = evaluate_checkpointed(suites, manifest, output_dir=output, run_binding=binding,
        env_factory=environment_factory, policy_factory=actor, progress=progress)
    end_identity = source_identity()
    if (end_identity["runtime_source_set_sha256"] != identity["runtime_source_set_sha256"]
        or end_identity["physical_model_sha256"] != identity["physical_model_sha256"]):
        raise ValueError("Runtime source or physical model changed during this run; no completion marker written")
    comparison = compare_baseline(payload, args.baseline_eval, args.policy)
    atomic_json(output/filename, payload)
    atomic_json(output/"comparison.json", comparison)
    overall = payload["suite_metrics"]["overall"]
    receipt = {"contract": CONTRACT, "run_binding_sha256": binding, "policy": args.policy,
        "episodes": len(payload["suite_metrics"]["episodes"]), "expected_episodes": EXPECTED_ROOTS,
        "coverage_complete": True, "source_stable": True, "checkpoint_counts_this_process": counts,
        "wall_seconds_this_process": time.perf_counter()-started,
        "truth_audited_task_successes": overall["truth_audited_task_success_episodes"],
        "truth_audited_task_failures": EXPECTED_ROOTS-overall["truth_audited_task_success_episodes"],
        "evaluation_file": filename, "evaluation_sha256": file_sha256(output/filename),
        "completion_semantics": "all frozen episodes evaluated and audited; individual task failures are preserved"}
    atomic_json(output/"run_receipt.json", receipt)
    atomic_json(completed_path, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
