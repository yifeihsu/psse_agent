"""Reviewed factories for frozen BC0 closed-loop release evaluation.

The evaluator fingerprints this whole module before it runs.  Keep the four
public factories small and deterministic, and pin the resulting source hash in
``bc0_evaluation_policy.json`` only after this module and its tests are frozen.

The learned-policy factory deliberately supports exactly two identities:

* the pinned base Gemma snapshot declared below; and
* a local PEFT adapter directory whose complete tree digest is supplied as the
  model revision.

There is no network, unpinned revision, raw-base fallback, or non-PEFT
checkpoint path.  One model bundle is cached per identity because the evaluator
constructs a fresh policy for every physical root, while loading a 31B model
must happen only once per evaluation process.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECTION_TOOLS,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    invalid_action,
    safe_normalize_action,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    validate_policy_payload,
)
from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import DEFAULT_CHI2_ALPHA
from psse_env.sft.gates import (
    GateError,
    _json_tool_call,
    _render,
    _tokenize_rendered,
    _validate_json_instance,
    parse_tool_call,
)
from psse_env.sft.training import infer_required_side_input_names
from psse_env.transactional_env import TransactionalPSSEEnv


EXPERT_POLICY_IDENTITY = "bc0-observable-handoff-expert-v2"
BASE_MODEL_ID = "unsloth/gemma-4-31B-it"
BASE_MODEL_REVISION = "8a796db4df380b178065ed910849477ff0e99c87"
# Exact repository tree at BASE_MODEL_REVISION. Git-managed files use their
# Git-blob SHA-1; LFS objects use the SHA-256 recorded by the immutable Hub
# commit. The release loader verifies every byte before Transformers sees it.
BASE_SNAPSHOT_FILE_MANIFEST: dict[str, tuple[int, str, str]] = {
    "chat_template.jinja": (16448, "git_blob_sha1", "98da08eb6be6f7a353d390456a6c4cacf811c9ed"),
    "config.json": (4702, "git_blob_sha1", "617506526283d9d31528363d58005c7bc5386bf3"),
    "generation_config.json": (208, "git_blob_sha1", "e605bb4523b1462ea9d9a3810b9e3ecf7ab7b1f6"),
    "model-00001-of-00002.safetensors": (
        49784788364,
        "sha256",
        "eeef8791537bc04f110967c513149e037d2a9ae97d49add7291ebfa62806bbfa",
    ),
    "model-00002-of-00002.safetensors": (
        12761549884,
        "sha256",
        "018912220f559f7025d60333e0996183cd538aa77ad6f4988a89ce47be681f10",
    ),
    "model.safetensors.index.json": (
        120246,
        "git_blob_sha1",
        "17fcef4a0ceb639c8ad4e2f7719f70aaff7de63b",
    ),
    "processor_config.json": (1689, "git_blob_sha1", "5465974d23e1eca2c46c2809b26c997946ce0d90"),
    "tokenizer.json": (
        32169626,
        "sha256",
        "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    ),
    "tokenizer_config.json": (19537, "git_blob_sha1", "f5082f7e7fba39a3fff443b1f874c75bf1dbe4e9"),
}
BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST: dict[str, tuple[int, str, str]] = {
    # Non-runtime repository documentation may be absent from a cache filled
    # through ``from_pretrained``. If present it is still verified and no
    # unknown extra file is accepted.
    ".gitattributes": (1708, "git_blob_sha1", "ddcb389f1c4efd0f6ec18e8925435134b04580da"),
    "README.md": (27903, "git_blob_sha1", "a86091ab51dacfc32a6997a909c3de413d999dec"),
}
MODEL_TREE_DIGEST_VERSION = "bc0-peft-tree-sha256-v1"
BC0_CHI2_ALPHA = DEFAULT_CHI2_ALPHA
# Training admission requires a 1.2 top-to-runner-up parameter ranking so one
# supervised target is unambiguous.  Closed-loop release evaluation instead
# preserves the frozen holdout's legacy rank-one inventory at 1.0: the expert
# may try ranked alternatives, but every proposed correction still has to pass
# the deployment candidate-quality gate before it can be committed.  Pin this
# explicitly so the release factory cannot silently inherit the provider's
# stricter single-label default and strand observably recoverable holdout roots.
BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD = 1.0
# Release evaluation uses the same bounded HIF search resolution as the
# reviewed round-0 aggregate builder.  The general provider defaults remain
# available outside this frozen release path, but a learner cannot turn one
# evaluation action into a 31 x 35 x 10 OpenDSS sweep.
BC0_HIF_ALPHA_GRID_SIZE = 5
BC0_HIF_R_GRID_SIZE = 7
BC0_HIF_MAX_SCANS = 3
# Frozen round-0 tool-call targets are at most 42 tokens. Keep a small
# malformed-output allowance, but fail boundedly when a checkpoint does not
# emit Gemma's pinned tool-response stop token.
MAX_NEW_TOKENS = 64
_REPO_ROOT = Path(__file__).resolve().parents[2]


def production_environment_factory(
    *, seed: int | None = None, rng: Any | None = None
) -> TransactionalPSSEEnv:
    """Construct the real MATPOWER-backed deployment environment.

    ``seed`` and ``rng`` are accepted only for the evaluator's uniform factory
    calling convention.  The provider stack is deterministic for a fixed
    scenario and does not draw from either value.
    """

    del seed, rng
    # The frozen suite and round-0 aggregate are validated at the same
    # significance level.  Relying on the provider's general-purpose default
    # made healthy release roots anomalous only at evaluation time.
    providers = MatpowerDeploymentProviders(
        chi2_alpha=BC0_CHI2_ALPHA,
        parameter_ranking_dominance_threshold=(
            BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
        hif_alpha_grid_size=BC0_HIF_ALPHA_GRID_SIZE,
        hif_r_grid_size=BC0_HIF_R_GRID_SIZE,
        hif_max_scans=BC0_HIF_MAX_SCANS,
    )
    env = TransactionalPSSEEnv(
        **providers.env_kwargs(),
        production_dataset_mode=True,
        max_steps=24,
        history_window=4,
    )
    if env.production_dataset_mode is not True:
        raise RuntimeError("release environment did not enter production dataset mode")
    if getattr(env.candidate_quality_oracle, "mode", None) != "deployment":
        raise RuntimeError("release environment candidate oracle is not in deployment mode")
    env.validate_production_configuration()
    return env


class ObservableExpertPolicy:
    """Policy-observation-only wrapper around the reviewed rule expert."""

    def __init__(self, expert: ExpertPolicyOracle) -> None:
        self._expert = expert

    @property
    def release_policy_identity(self) -> dict[str, str | None]:
        """Return a fresh identity mapping so callers cannot mutate it in place."""

        return {
            "explicit_policy_identity": EXPERT_POLICY_IDENTITY,
            "model_id": None,
            "model_revision": None,
        }

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        selection = select_observable_expert_actions(
            policy_observation=observation, expert_oracle=self._expert
        )
        return (
            copy.deepcopy(selection.preferred_action)
            if selection.preferred_action is not None
            else invalid_action("observable_expert_returned_no_action")
        )


@dataclass(frozen=True)
class ObservableExpertSelection:
    """One observable expert decision, with the evidence it was made from."""

    actions: tuple[dict[str, Any], ...]
    preferred_action: dict[str, Any] | None
    bounded_history: tuple[dict[str, Any], ...]
    #: "candidate_disposition_reconstruction" or "rule_expert".
    selection_basis: str


def select_observable_expert_actions(
    *,
    policy_observation: Mapping[str, Any],
    expert_oracle: Any,
) -> ObservableExpertSelection:
    """The single observable expert-selection path for DAgger-1.

    Three callers previously reimplemented this: the release policy wrapper, the
    natural rollout collector, and the recovery-probe generator.  The probe
    generator's copy called the rule expert directly with the driver's full
    history, which diverged in two ways that mattered.  The rule expert cannot
    decide commit or rollback from a policy observation alone, so a probe
    episode stalled at a verified candidate; and the unbounded history gave the
    expert evidence the learner could not see.

    The whole ordered list is returned, not just the first action: the rank-one
    proof needs the remainder to establish that the target is observably first.

    History is read only from ``history_window``.  A caller's longer private
    history is never substituted, and a malformed window fails closed rather
    than silently falling back to it.
    """

    if not isinstance(policy_observation, Mapping):
        raise TypeError("observable expert requires a policy-observation mapping")
    observation = copy.deepcopy(dict(policy_observation))
    # Reject, rather than strip, a caller that crosses the policy/oracle
    # boundary, so an OracleState or hidden truth can never reach the expert.
    validate_policy_payload(observation)
    raw_history = observation.get("history_window") or []
    if not isinstance(raw_history, list) or any(
        not isinstance(item, Mapping) for item in raw_history
    ):
        raise ValueError("policy observation history_window must contain mappings")
    bounded_history = [copy.deepcopy(dict(item)) for item in raw_history]

    disposition_action = _observable_candidate_disposition_action(
        observation, bounded_history
    )
    if disposition_action is not None:
        action = safe_normalize_action(disposition_action)
        actions = (
            () if action["tool"] == "__invalid_action__" else (copy.deepcopy(action),)
        )
        return ObservableExpertSelection(
            actions=actions,
            preferred_action=copy.deepcopy(actions[0]) if actions else None,
            bounded_history=tuple(bounded_history),
            selection_basis="candidate_disposition_reconstruction",
        )

    raw_actions = expert_oracle.next_actions(
        observation, [copy.deepcopy(dict(item)) for item in bounded_history]
    )
    normalized = tuple(
        action
        for action in (safe_normalize_action(item) for item in raw_actions)
        if action["tool"] != "__invalid_action__"
    )
    return ObservableExpertSelection(
        actions=normalized,
        preferred_action=copy.deepcopy(normalized[0]) if normalized else None,
        bounded_history=tuple(bounded_history),
        selection_basis="rule_expert",
    )


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _observable_verification_source(value: Any) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized or any(
        token in normalized
        for token in ("hidden", "oracle", "truth", "synthetic", "placeholder", "fallback")
    ):
        return False
    return normalized.startswith(
        (
            "observable",
            "deployment",
            "sensor",
            "wls",
            "configured_provider",
            "real",
            "production",
        )
    )


def _latest_candidate_correction(
    history: list[Mapping[str, Any]], candidate_id: str
) -> dict[str, Any] | None:
    for event in reversed(history):
        action = safe_normalize_action(event.get("action") or {})
        if action["tool"] not in CORRECTION_TOOLS:
            continue
        output = event.get("tool_output")
        if not isinstance(output, Mapping) or output.get("execution_status") != "success":
            return None
        observed_candidate = output.get("candidate_state_id")
        if observed_candidate is not None and str(observed_candidate) != candidate_id:
            return None
        return action
    return None


def _observable_candidate_disposition(
    observation: Mapping[str, Any], history: list[Mapping[str, Any]]
) -> str | None:
    """Reconstruct the deployment decision from policy-visible metrics only."""

    if observation.get("has_verified_candidate") is not True:
        return None
    candidate_id = str(observation.get("candidate_state_id") or "").strip()
    verification = observation.get("last_verification")
    if not candidate_id or not isinstance(verification, Mapping):
        return "inconclusive"
    if verification.get("execution_status", "success") != "success":
        return "inconclusive"
    if str(verification.get("state_id") or "") != candidate_id:
        return "inconclusive"
    if not _observable_verification_source(verification.get("evidence_source")):
        return "inconclusive"

    correction = _latest_candidate_correction(history, candidate_id)
    if correction is None:
        return "inconclusive"
    family = {
        "correct_measurements": "measurement",
        "correct_parameters": "parameter",
        "correct_topology": "topology",
    }.get(correction["tool"])
    if family is None:
        return "inconclusive"

    physical = verification.get("physical_constraints_ok")
    feasibility = [
        verification.get(key)
        for key in ("power_flow_converged", "topology_feasible")
        if verification.get(key) is not None
    ]
    violations = verification.get(
        "physical_bound_violations",
        verification.get(
            "new_constraint_violations", verification.get("new_violations")
        ),
    )
    if isinstance(violations, list):
        violation_count = len(violations)
    else:
        numeric_violations = _finite_float(violations)
        violation_count = int(numeric_violations or 0.0)
    if physical is False or any(value is False for value in feasibility) or violation_count > 0:
        return "rollback"
    physical_known_safe = physical is True or (
        any(value is True for value in feasibility) and violations is not None
    )
    if not physical_known_safe:
        return "inconclusive"

    if (
        family == "measurement"
        and verification.get("sequential_cross_family_measurement") is True
        and verification.get("measurement_evidence_dominant") is not True
    ):
        locality = verification.get("measurement_target_branch_colocated")
        if locality is None:
            return "inconclusive"
        if locality is True:
            return "rollback"

    target_progress = _finite_float(verification.get("target_progress"))
    global_progress = _finite_float(verification.get("global_progress"))
    target_metric = _finite_float(verification.get("target_metric_value"))
    target_threshold = _finite_float(verification.get("target_metric_threshold"))
    target_cleared = bool(
        target_metric is not None
        and target_threshold is not None
        and target_threshold >= 0.0
        and target_metric < target_threshold
    )

    global_resolved_value = verification.get(
        "globally_resolved", verification.get("post_action_resolved")
    )
    if global_resolved_value is not None:
        global_resolved: bool | None = bool(global_resolved_value)
    else:
        remaining_score = _finite_float(verification.get("remaining_anomaly_score"))
        anomaly_threshold = _finite_float(
            verification.get(
                "anomaly_threshold", verification.get("chi_square_threshold")
            )
        )
        global_resolved = (
            remaining_score < anomaly_threshold
            if remaining_score is not None and anomaly_threshold is not None
            else None
        )

    if family == "topology" and verification.get(
        "topology_target_status_matches_requested"
    ) is True:
        multiplier = _finite_float(
            verification.get("topology_target_branch_multiplier")
        )
        multiplier_threshold = _finite_float(
            verification.get("topology_target_branch_multiplier_threshold")
        )
        if (
            multiplier is not None
            and multiplier_threshold is not None
            and multiplier_threshold > 0.0
            and multiplier > 1.25 * multiplier_threshold
            and (global_progress is None or global_progress < 0.95)
        ):
            return "rollback"

    partial_floor = 0.30 if family in {"parameter", "topology"} else 0.20
    if target_cleared:
        if global_resolved is True:
            return "commit"
        if global_resolved is False:
            return (
                "commit"
                if global_progress is not None and global_progress >= partial_floor
                else "rollback"
            )
        return "inconclusive"

    branch_material_progress = bool(
        family in {"parameter", "topology"}
        and target_progress is not None
        and target_progress >= 0.80
        and target_metric is not None
        and target_threshold is not None
        and target_threshold > 0.0
        and target_metric <= 1.25 * target_threshold
        and global_progress is not None
        and global_progress >= 0.50
    )
    if branch_material_progress and global_resolved is False:
        return "commit"

    # Deployment verification exposes an explicit target metric and threshold.
    # When that target is still at or above its threshold, the deployment
    # CandidateQualityOracle rejects the candidate unless a parameter/topology
    # branch satisfies the narrowly bounded material-progress exception above.
    # Falling through to the generic positive-progress rule used to turn any
    # small improvement into a commit, even though the environment would
    # (correctly) reject that same candidate.  Keep the observable
    # reconstruction aligned with the deployment gate and fail closed.
    if (
        target_metric is not None
        and target_threshold is not None
        and target_threshold >= 0.0
        and target_metric >= target_threshold
    ):
        return "rollback"

    if target_progress is not None and target_progress < -0.05:
        return "rollback"
    target_improved = target_progress is not None and target_progress >= 0.05
    if target_improved and global_resolved is True:
        return "commit"
    if target_improved and global_resolved is False:
        return "commit" if global_progress is None or global_progress >= 0.0 else "rollback"
    if target_improved:
        return "inconclusive"
    return "inconclusive" if target_progress is None or abs(target_progress) < 0.05 else "rollback"


def _observable_candidate_disposition_action(
    observation: Mapping[str, Any], history: list[Mapping[str, Any]]
) -> dict[str, Any] | None:
    candidate_id = str(observation.get("candidate_state_id") or "").strip()
    # A structured process failure is newer and more authoritative than the
    # verification metrics that preceded it.  In particular, after a rejected
    # commit the candidate remains open and verified, so recomputing only from
    # stale verification evidence previously repeated commit_state until the
    # step budget expired.  Consume the policy-visible repair action emitted
    # by the environment before reconstructing a disposition.
    last_output = observation.get("last_tool_output")
    if (
        candidate_id
        and observation.get("last_tool") == COMMIT_STATE
        and isinstance(last_output, Mapping)
        and last_output.get("execution_status") == "failure"
    ):
        for raw_action in last_output.get("valid_next_actions") or []:
            if not isinstance(raw_action, Mapping):
                continue
            repair = safe_normalize_action(raw_action)
            if (
                repair["tool"] == ROLLBACK_STATE
                and str(repair["arguments"].get("candidate_state_id") or "")
                == candidate_id
            ):
                return repair

    disposition = _observable_candidate_disposition(observation, history)
    if disposition is None:
        return None
    if not candidate_id:
        return invalid_action("verified_candidate_id_missing")
    if disposition == "commit":
        return {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}
    if disposition == "rollback" or observation.get("last_tool") in {
        ASK_FOR_MORE_EVIDENCE,
        RUN_ALTERNATIVE_TEST,
    }:
        return {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}
    return {
        "tool": ASK_FOR_MORE_EVIDENCE,
        "arguments": {"state_id": candidate_id},
    }


def observable_expert_policy_factory(
    *,
    policy_identity: str | None = None,
    seed: int | None = None,
    rng: Any | None = None,
) -> ObservableExpertPolicy:
    """Build the immutable BC0 observable expert comparator."""

    del seed, rng
    requested = str(policy_identity or "").strip()
    if requested != EXPERT_POLICY_IDENTITY:
        raise ValueError(
            "observable expert requires policy_identity="
            f"{EXPERT_POLICY_IDENTITY!r}, got {requested!r}"
        )
    # Match the deployment environment's hydrated-correction process contract,
    # without supplying the expert any environment or oracle state.
    process_oracle = ProcessValidityOracle(executor_hydrated_corrections=True)
    return ObservableExpertPolicy(ExpertPolicyOracle(process_oracle=process_oracle))


def deterministic_case_loader(value: Any) -> dict[str, Any]:
    """Load one MATPOWER case through the production parser.

    The wrapper accepts the two representations emitted by the deployment
    stack: a path-like value, or a mapping containing ``case_path``.  Relative
    references are resolved against the source root, never the caller's
    working directory; the production parser remains the single
    physical-comparison implementation.
    """

    if isinstance(value, Mapping):
        unknown = sorted(str(key) for key in value if str(key) != "case_path")
        if unknown:
            raise ValueError(f"case loader mapping has unsupported fields: {unknown}")
        value = value.get("case_path")
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if not isinstance(value, str) or not value.strip():
        raise TypeError("case loader requires a non-empty path or {case_path: path}")

    provided = value.strip()
    if provided == "case14":
        case_path = _REPO_ROOT / "mcp_server" / "case14.m"
    else:
        case_path = Path(provided).expanduser()
        if not case_path.is_absolute():
            case_path = _REPO_ROOT / case_path
    try:
        case_path = case_path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise FileNotFoundError(f"MATPOWER case does not exist: {provided}") from exc
    if not case_path.is_file():
        raise ValueError(f"MATPOWER case is not a regular file: {case_path}")

    from mcp_server.matpower_server import _load_python_case

    loaded = _load_python_case(str(case_path))
    if not isinstance(loaded, Mapping):
        raise TypeError("production MATPOWER parser did not return a mapping")
    required = {"baseMVA", "bus", "gen", "branch"}
    missing = sorted(required - set(loaded))
    if missing:
        raise ValueError(f"parsed MATPOWER case is missing fields: {missing}")
    return dict(loaded)


def _reject_symlink_path_components(path: Path) -> None:
    """Reject a symlink at the root or in any existing parent component."""

    absolute = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            # The caller reports a more specific missing-root error.
            return
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"PEFT checkpoint path contains a symlink: {current}")


def _checkpoint_files(root: Path) -> list[Path]:
    _reject_symlink_path_components(root)
    if not root.is_dir():
        raise ValueError(f"PEFT checkpoint is not a directory: {root}")
    files: list[Path] = []
    for item in root.rglob("*"):
        relative = item.relative_to(root).as_posix()
        metadata = item.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"PEFT checkpoint contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"PEFT checkpoint contains a non-regular entry: {relative}"
            )
        if metadata.st_nlink != 1:
            raise ValueError(
                f"PEFT checkpoint contains a multiply linked file: {relative}"
            )
        files.append(item)
    if not files:
        raise ValueError("PEFT checkpoint directory is empty")
    return sorted(files, key=lambda entry: entry.relative_to(root).as_posix())


def _open_checkpoint_file(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        os.close(descriptor)
        raise ValueError(f"PEFT checkpoint file is not an unlinked regular file: {path}")
    return descriptor, metadata


def _update_tree_digest(
    digest: Any, root: Path, item: Path, *, copy_to: Path | None = None
) -> None:
    relative = item.relative_to(root).as_posix()
    descriptor, before = _open_checkpoint_file(item)
    digest.update(relative.encode("utf-8") + b"\0" + str(before.st_size).encode("ascii") + b"\0")
    destination = None
    try:
        if copy_to is not None:
            target = copy_to / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            destination = target.open("xb")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                if destination is not None:
                    destination.write(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"PEFT checkpoint changed while reading: {relative}")
    finally:
        if destination is not None:
            destination.close()
        os.close(descriptor)
    digest.update(b"\0")


def checkpoint_tree_sha256(path: str | os.PathLike[str]) -> str:
    """Return the canonical digest of a local PEFT checkpoint tree.

    The digest covers every relative file name, byte length, and byte.  File
    modes and mtimes are intentionally excluded so a byte-identical transfer to
    HPC retains its identity.  Symlinks and non-regular filesystem entries are
    rejected to prevent the digest from naming bytes outside the checkpoint.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    files = _checkpoint_files(root)

    digest = hashlib.sha256()
    digest.update(MODEL_TREE_DIGEST_VERSION.encode("ascii") + b"\0")
    for item in files:
        _update_tree_digest(digest, root, item)
    return digest.hexdigest()


def inspect_release_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Validate an adapter-only release input and report its stable identity.

    This is a no-copy launcher preflight.  The model factory still repeats the
    digest while copying into its private tree, so a later mutation cannot
    change the loaded bytes under this identity.
    """

    checkpoint = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    files = _checkpoint_files(checkpoint)
    _validate_adapter_contents(checkpoint)
    return {
        "path": str(checkpoint),
        "tree_sha256": checkpoint_tree_sha256(checkpoint),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
    }


def _copy_verified_checkpoint_tree(
    path: str | os.PathLike[str], revision: str
) -> tuple[Path, tempfile.TemporaryDirectory[str]]:
    """Copy exactly the named bytes to a private tree and verify that copy.

    PEFT receives only the private tree.  Thus a mutation of the user-visible
    checkpoint during or after copying can never change the bytes loaded under
    the attested digest.
    """

    root = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    files = _checkpoint_files(root)
    owner = tempfile.TemporaryDirectory(prefix="bc0_verified_peft_")
    snapshot = Path(owner.name) / "adapter"
    snapshot.mkdir()
    digest = hashlib.sha256()
    digest.update(MODEL_TREE_DIGEST_VERSION.encode("ascii") + b"\0")
    try:
        for item in files:
            _update_tree_digest(digest, root, item, copy_to=snapshot)
        actual = digest.hexdigest()
        if actual != revision.lower():
            raise GateError(
                "Local PEFT checkpoint tree digest mismatch: "
                f"expected {revision.lower()}, computed {actual}"
            )
        # Verify the exact tree PEFT will read, independently of the source.
        if checkpoint_tree_sha256(snapshot) != revision.lower():
            raise GateError("Private PEFT checkpoint copy failed identity verification")
        for item in _checkpoint_files(snapshot):
            item.chmod(0o400)
        for directory in sorted(
            (item for item in snapshot.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            directory.chmod(0o500)
        snapshot.chmod(0o500)
    except Exception:
        owner.cleanup()
        raise
    return snapshot, owner


@dataclass(frozen=True)
class _ModelBundle:
    model: Any
    processor: Any
    model_id: str
    model_revision: str
    adapter_snapshot_path: str | None = None
    adapter_snapshot_owner: Any | None = None


_MODEL_BUNDLES: dict[tuple[str, str], _ModelBundle] = {}
_MODEL_BUNDLE_LOCK = threading.Lock()


def _verify_snapshot_tree(
    snapshot: Path,
    expected_manifest: Mapping[str, tuple[int, str, str]],
    optional_manifest: Mapping[str, tuple[int, str, str]] | None = None,
) -> None:
    """Verify an exact local Hub snapshot against immutable object hashes."""

    observed = {
        item.relative_to(snapshot).as_posix()
        for item in snapshot.rglob("*")
        if item.is_file() or item.is_symlink()
    }
    optional_manifest = optional_manifest or {}
    expected = set(expected_manifest)
    allowed = expected | set(optional_manifest)
    missing = sorted(expected - observed)
    unexpected = sorted(observed - allowed)
    if missing or unexpected:
        raise GateError(
            "Pinned Gemma snapshot file manifest mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )

    manifest = {**optional_manifest, **expected_manifest}
    for relative in sorted(observed):
        expected_size, algorithm, expected_digest = manifest[relative]
        path = snapshot / relative
        before_link = path.lstat()
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise GateError(
                f"Pinned Gemma snapshot entry cannot be resolved: {relative}"
            ) from exc
        descriptor = os.open(
            resolved,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
                raise GateError(
                    "Pinned Gemma snapshot entry has the wrong type or size: "
                    f"{relative}"
                )
            if algorithm == "sha256":
                digest = hashlib.sha256()
            elif algorithm == "git_blob_sha1":
                digest = hashlib.sha1()
                digest.update(f"blob {expected_size}\0".encode("ascii"))
            else:  # pragma: no cover - constant construction error
                raise RuntimeError(f"unsupported snapshot digest algorithm: {algorithm}")
            with os.fdopen(descriptor, "rb", closefd=False) as handle:
                for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                    digest.update(block)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        after_link = path.lstat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
            before_link.st_dev,
            before_link.st_ino,
            before_link.st_mtime_ns,
            before_link.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
            after_link.st_dev,
            after_link.st_ino,
            after_link.st_mtime_ns,
            after_link.st_ctime_ns,
        ):
            raise GateError(
                f"Pinned Gemma snapshot changed while hashing: {relative}"
            )
        if digest.hexdigest() != expected_digest:
            raise GateError(
                f"Pinned Gemma snapshot object digest mismatch: {relative}"
            )


def _resolve_base_snapshot() -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - live optional dependency
        raise GateError(f"huggingface_hub is required for release inference: {exc}") from exc

    try:
        raw_path = snapshot_download(
            repo_id=BASE_MODEL_ID,
            revision=BASE_MODEL_REVISION,
            local_files_only=True,
        )
    except Exception as exc:  # pragma: no cover - live cache state
        raise GateError(
            "Pinned base snapshot is absent from the local Hugging Face cache: "
            f"{BASE_MODEL_ID}@{BASE_MODEL_REVISION}: {type(exc).__name__}: {exc}"
        ) from exc
    snapshot = Path(raw_path).expanduser().resolve(strict=True)
    if not snapshot.is_dir() or snapshot.name.lower() != BASE_MODEL_REVISION:
        raise GateError(
            "Hugging Face cache did not resolve to the exact pinned snapshot: "
            f"{snapshot}"
        )
    _verify_snapshot_tree(
        snapshot,
        BASE_SNAPSHOT_FILE_MANIFEST,
        BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST,
    )
    config_path = snapshot / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"Pinned Gemma config is unavailable or invalid: {exc}") from exc
    if config.get("model_type") != "gemma4" or config.get("architectures") != [
        "Gemma4ForConditionalGeneration"
    ]:
        raise GateError("Pinned snapshot is not the reviewed Gemma 4 conditional model")
    return snapshot


def _object_source_path(value: Any) -> str | None:
    for candidate in (value, getattr(value, "tokenizer", None)):
        if candidate is None:
            continue
        for attribute in ("name_or_path", "_name_or_path"):
            raw = getattr(candidate, attribute, None)
            if isinstance(raw, str) and raw.strip():
                return raw.strip()
    return None


def _require_loaded_from_snapshot(value: Any, snapshot: Path, *, label: str) -> None:
    source = _object_source_path(value)
    if source is None:
        raise GateError(f"Loaded {label} exposes no source path for identity attestation")
    try:
        actual = Path(source).expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GateError(f"Loaded {label} source path cannot be resolved: {source!r}") from exc
    if actual != snapshot:
        raise GateError(
            f"Loaded {label} came from {actual}, expected pinned snapshot {snapshot}"
        )


def _load_base_components() -> tuple[Any, Any]:
    """Load the exact local snapshot through one Transformers model class."""

    snapshot = _resolve_base_snapshot()
    try:
        import torch
        from transformers import (
            AutoModelForImageTextToText,
            AutoProcessor,
            BitsAndBytesConfig,
        )
    except Exception as exc:  # pragma: no cover - live optional dependencies
        raise GateError(f"Transformers release dependencies are unavailable: {exc}") from exc

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": "auto",
        "quantization_config": quantization,
        "dtype": torch.bfloat16,
    }
    try:
        processor = AutoProcessor.from_pretrained(
            str(snapshot),
            local_files_only=True,
            trust_remote_code=False,
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(snapshot), **load_kwargs
        )
    except Exception as exc:  # pragma: no cover - live model/cache state
        raise GateError(
            "Exact pinned Gemma processor/model load failed; no loader fallback was used: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not callable(getattr(processor, "apply_chat_template", None)):
        raise GateError("Pinned processor exposes no apply_chat_template")
    decoder = (
        processor
        if callable(getattr(processor, "decode", None))
        else getattr(processor, "tokenizer", None)
    )
    if decoder is None or not callable(getattr(decoder, "decode", None)):
        raise GateError("Pinned processor exposes no decoder")
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "gemma4":
        raise GateError("Loaded model config is not Gemma 4")
    _require_loaded_from_snapshot(config, snapshot, label="model")
    _require_loaded_from_snapshot(processor, snapshot, label="processor")
    # The shared Hub cache was verified before either loader opened it.  Hash
    # the exact tree again after both reads so a concurrent replacement or
    # mutation cannot leave a loaded model carrying only a pre-load path
    # attestation.
    _verify_snapshot_tree(
        snapshot,
        BASE_SNAPSHOT_FILE_MANIFEST,
        BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST,
    )
    model.eval()
    return model, processor


def _validate_adapter_contents(checkpoint: Path) -> None:
    config_path = checkpoint / "adapter_config.json"
    weight_paths = (
        checkpoint / "adapter_model.safetensors",
        checkpoint / "adapter_model.bin",
    )
    if not config_path.is_file() or not any(path.is_file() for path in weight_paths):
        raise GateError(
            "Local checkpoint must contain adapter_config.json and PEFT adapter weights"
        )
    try:
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"PEFT adapter_config.json is invalid: {exc}") from exc
    if adapter_config.get("base_model_name_or_path") != BASE_MODEL_ID:
        raise GateError(
            "PEFT adapter base_model_name_or_path does not match the pinned base model"
        )
    peft_type = str(adapter_config.get("peft_type") or "").upper()
    if peft_type != "LORA":
        raise GateError(f"Release checkpoint must be a LoRA PEFT adapter, got {peft_type!r}")


def _validate_adapter_tree(path: str, revision: str) -> Path:
    checkpoint = Path(os.path.abspath(os.path.expanduser(path)))
    actual_digest = checkpoint_tree_sha256(checkpoint)
    if revision.lower() != actual_digest:
        raise GateError(
            "Local PEFT checkpoint tree digest mismatch: "
            f"expected {revision.lower()}, computed {actual_digest}"
        )
    _validate_adapter_contents(checkpoint)
    return checkpoint


def _load_model_bundle(model_id: str, model_revision: str) -> _ModelBundle:
    adapter_snapshot_path: str | None = None
    adapter_snapshot_owner: Any | None = None
    if model_id == BASE_MODEL_ID:
        if model_revision != BASE_MODEL_REVISION:
            raise GateError(
                f"Base model revision must be exactly {BASE_MODEL_REVISION}"
            )
        model, processor = _load_base_components()
    else:
        if len(model_revision) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in model_revision
        ):
            raise GateError("Local PEFT checkpoint revision must be a 64-hex tree digest")
        if not Path(model_id).expanduser().is_absolute():
            raise GateError("Local PEFT checkpoint model_id must be an absolute path")
        try:
            checkpoint, adapter_snapshot_owner = _copy_verified_checkpoint_tree(
                model_id, model_revision
            )
            adapter_snapshot_path = str(checkpoint)
            _validate_adapter_contents(checkpoint)
            model, processor = _load_base_components()
            try:
                from peft import PeftModel
            except Exception as exc:  # pragma: no cover - live optional dependency
                raise GateError(f"PEFT is required for checkpoint evaluation: {exc}") from exc
            try:
                model = PeftModel.from_pretrained(
                    model,
                    str(checkpoint),
                    is_trainable=False,
                    local_files_only=True,
                )
            except Exception as exc:  # pragma: no cover - live checkpoint state
                raise GateError(
                    "Exact local PEFT adapter load failed; the raw base model was not used: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            peft_config = getattr(model, "peft_config", None)
            if not isinstance(peft_config, Mapping) or not peft_config:
                raise GateError("PEFT loader returned a model with no active adapter config")
            if checkpoint_tree_sha256(checkpoint) != model_revision.lower():
                raise GateError("Private PEFT checkpoint changed while it was being loaded")
            model.eval()
        except Exception:
            if adapter_snapshot_owner is not None:
                adapter_snapshot_owner.cleanup()
            raise
    return _ModelBundle(
        model=model,
        processor=processor,
        model_id=model_id,
        model_revision=model_revision.lower(),
        adapter_snapshot_path=adapter_snapshot_path,
        adapter_snapshot_owner=adapter_snapshot_owner,
    )


def _cached_model_bundle(model_id: str, model_revision: str) -> _ModelBundle:
    key = (model_id, model_revision.lower())
    with _MODEL_BUNDLE_LOCK:
        bundle = _MODEL_BUNDLES.get(key)
        if bundle is None:
            bundle = _load_model_bundle(*key)
            _MODEL_BUNDLES[key] = bundle
        return bundle


def _model_input_device(model: Any) -> Any:
    try:
        embeddings = model.get_input_embeddings()
        return next(embeddings.parameters()).device
    except (AttributeError, StopIteration):
        try:
            return next(model.parameters()).device
        except StopIteration as exc:
            raise GateError("Loaded release model has no parameters") from exc


_NATIVE_TOOL_CALL_RE = re.compile(
    r"(?:<\|tool_call\|>|<tool_call>)?\s*call:([A-Za-z_][A-Za-z0-9_.-]*)"
)


def _json_tool_call_occurrences(text: str) -> int:
    """Count top-level JSON tool-call objects without double-counting nesting."""

    decoder = json.JSONDecoder()
    records: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if _json_tool_call(value) is not None:
            records.append((index, end))
    top_level = [
        record
        for record in records
        if not any(
            outer != record
            and outer[0] <= record[0]
            and record[1] <= outer[1]
            for outer in records
        )
    ]
    return len(top_level)


def _parse_exact_tool_call(text: str) -> Any:
    native_calls = len(_NATIVE_TOOL_CALL_RE.findall(text))
    json_calls = _json_tool_call_occurrences(text)
    if native_calls + json_calls != 1:
        raise GateError(
            "Generated output must contain exactly one tool call; found "
            f"{native_calls + json_calls}"
        )
    return parse_tool_call(text)


def _validated_generated_action(
    text: str, parameter_schemas: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    parsed = _parse_exact_tool_call(text)
    parameters = parameter_schemas.get(parsed.name)
    if not isinstance(parameters, Mapping):
        raise GateError(f"Generated tool {parsed.name!r} is not in the pinned registry")
    properties = parameters.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    unsupported = sorted(set(parsed.arguments) - set(properties))
    if unsupported:
        raise GateError(
            f"Generated tool {parsed.name!r} has unsupported arguments: {unsupported}"
        )
    _validate_json_instance(
        parsed.arguments,
        parameters,
        path=f"generated.{parsed.name}.arguments",
    )
    return {"tool": parsed.name, "arguments": copy.deepcopy(parsed.arguments)}


class _CanonicalGemmaPolicy:
    """Generate exactly one canonical tool call using the SFT render path."""

    def __init__(self, bundle: _ModelBundle) -> None:
        self._bundle = bundle
        self._tools = unified_tool_schemas()
        self._parameter_schemas = {
            str(row["function"]["name"]): row["function"]["parameters"]
            for row in self._tools
        }
        self._last_action_metrics: dict[str, Any] = {}

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        """Return non-semantic inference telemetry for release progress logs."""

        return copy.deepcopy(self._last_action_metrics)

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._last_action_metrics = {}
        if not isinstance(observation, Mapping):
            raise TypeError("Gemma policy requires a model-observation mapping")
        payload = {"state": copy.deepcopy(dict(observation))}
        validate_policy_payload(payload)
        messages = [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, allow_nan=False),
            },
        ]
        rendered = _render(
            self._bundle.processor,
            messages,
            self._tools,
            add_generation_prompt=True,
        )
        encoded = _tokenize_rendered(self._bundle.processor, rendered)
        prompt_length = len(encoded["input_ids"])
        if prompt_length <= 0:
            raise GateError("Pinned processor produced an empty release prompt")
        required_side_inputs = infer_required_side_input_names(
            self._bundle.model,
            self._bundle.processor,
            BASE_MODEL_ID,
        )
        for name in required_side_inputs:
            values = encoded.setdefault(name, [0] * prompt_length)
            if len(values) != prompt_length:
                raise GateError(f"Pinned processor produced unaligned {name}")

        try:
            import torch
        except Exception as exc:  # pragma: no cover - live optional dependency
            raise GateError(f"torch is required for release generation: {exc}") from exc
        device = _model_input_device(self._bundle.model)
        inputs = {
            key: torch.tensor([values], dtype=torch.long, device=device)
            for key, values in encoded.items()
            if key == "input_ids"
            or key == "attention_mask"
            or key.endswith("token_type_ids")
        }
        self._bundle.model.eval()
        generation_started = time.perf_counter()
        with torch.inference_mode():
            generated = self._bundle.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )
        generation_seconds = time.perf_counter() - generation_started
        output_ids = generated[0][prompt_length:].detach().cpu()
        generated_tokens = int(output_ids.numel())
        self._last_action_metrics = {
            "prompt_tokens": int(prompt_length),
            "generated_tokens": generated_tokens,
            "generation_seconds": float(generation_seconds),
            "hit_max_new_tokens": generated_tokens >= MAX_NEW_TOKENS,
            "last_token_id": (
                int(output_ids[-1].item()) if generated_tokens else None
            ),
        }
        decoder = (
            self._bundle.processor
            if callable(getattr(self._bundle.processor, "decode", None))
            else self._bundle.processor.tokenizer
        )
        text = decoder.decode(output_ids, skip_special_tokens=False)
        return _validated_generated_action(text, self._parameter_schemas)


class GemmaReleasePolicy:
    """Canonical-generation policy with controller alias binding and identity."""

    def __init__(self, bundle: _ModelBundle) -> None:
        self._canonical_policy = _CanonicalGemmaPolicy(bundle)
        self._adapter = LocalAliasPolicyAdapter(
            self._canonical_policy, protocol="canonical"
        )
        self._model_id = bundle.model_id
        self._model_revision = bundle.model_revision

    @property
    def release_policy_identity(self) -> dict[str, str | None]:
        """Return the exact loaded identity without exposing mutable state."""

        return {
            "explicit_policy_identity": None,
            "model_id": self._model_id,
            "model_revision": self._model_revision,
        }

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        """Expose timing/token counts without exposing model or prompt state."""

        return self._canonical_policy.last_action_metrics

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        validate_policy_payload(observation)
        return self._adapter.act(copy.deepcopy(dict(observation)))


def gemma_release_policy_factory(
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    seed: int | None = None,
    rng: Any | None = None,
) -> GemmaReleasePolicy:
    """Build a pinned base-Gemma or content-addressed local-PEFT policy."""

    del seed, rng
    normalized_id = str(model_id or "").strip()
    normalized_revision = str(model_revision or "").strip().lower()
    if not normalized_id or not normalized_revision:
        raise ValueError("Gemma release policy requires model_id and model_revision")
    bundle = _cached_model_bundle(normalized_id, normalized_revision)
    if (
        bundle.model_id != normalized_id
        or bundle.model_revision != normalized_revision
    ):
        raise RuntimeError("cached model bundle identity does not match the requested identity")
    return GemmaReleasePolicy(bundle)


__all__ = [
    "BASE_MODEL_ID",
    "BASE_MODEL_REVISION",
    "BC0_CHI2_ALPHA",
    "EXPERT_POLICY_IDENTITY",
    "GemmaReleasePolicy",
    "ObservableExpertPolicy",
    "checkpoint_tree_sha256",
    "deterministic_case_loader",
    "gemma_release_policy_factory",
    "inspect_release_checkpoint",
    "observable_expert_policy_factory",
    "production_environment_factory",
]
