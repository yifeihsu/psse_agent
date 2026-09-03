from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from collections.abc import Mapping

from psse_env.dagger import collect_dagger1 as c
from psse_env.sft.provenance import file_sha256, git_source_state

repo = Path(os.environ["REPO"]).resolve()
d0 = Path(os.environ["D0_DIR"]).resolve()
d1 = Path(os.environ["D1_DIR"]).resolve()
learner = Path(os.environ["LEARNER"]).resolve()
commit = os.environ["COMMIT"]
revision = os.environ["REV"]
receipt_path = Path(os.environ["LOG_ROOT"]) / "input_audit.json"

source = git_source_state(repo)
if source.get("release_eligible_source") is not True or source.get("source_commit") != commit:
    raise RuntimeError("source checkout is not clean and bound to the reviewed commit")

input_path = d1 / "scenarios.json"
report_path = d1 / "scenario_generator_report.json"
manifest_path = d1 / "scenarios.json.manifest.json"
holdout_path = d1 / "development_holdout.json"
holdout_manifest_path = d1 / "development_holdout.json.manifest.json"
holdout_report_path = d1 / "development_holdout.generator.json"
for path in (input_path, report_path, manifest_path, holdout_path, holdout_manifest_path, holdout_report_path):
    if not path.is_file():
        raise FileNotFoundError(path)

scenarios = c._load_json_or_jsonl(input_path)
report = json.loads(report_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(report, Mapping) or not isinstance(manifest, Mapping):
    raise TypeError("scenario report/manifest must be JSON objects")
c.validate_training_source_report(report)

frozen_suite = c.DEFAULT_FORBIDDEN_SUITE
policy_path = c.DEFAULT_EVALUATION_POLICY
frozen_roots = c.frozen_physical_roots(frozen_suite)
suite_sha = file_sha256(frozen_suite)
policy = json.loads(policy_path.read_text(encoding="utf-8"))
suite_policy = policy.get("suite_policy") if isinstance(policy, Mapping) else None
if not isinstance(suite_policy, Mapping):
    raise RuntimeError("evaluation policy has no suite policy")
if suite_policy.get("status") != "pinned" or suite_policy.get("approved_suite_sha256") != suite_sha:
    raise RuntimeError("frozen evaluation suite does not match its policy pin")

d0_raw = d0 / "aggregate.raw.jsonl"
d0_prov_path = d0 / "aggregate.generation_provenance.json"
d0_manifest = d0 / c.AGGREGATE_MANIFEST_FILENAME
for path in (d0_raw, d0_prov_path, d0_manifest):
    if not path.is_file():
        raise FileNotFoundError(path)
d0_rows = c._load_json_or_jsonl(d0_raw)
d0_roots = frozenset(str(row.get("physical_root_fingerprint") or "").strip() for row in d0_rows if str(row.get("physical_root_fingerprint") or "").strip())
if not d0_roots:
    raise RuntimeError("D0 aggregate has no physical roots")
d0_prov = json.loads(d0_prov_path.read_text(encoding="utf-8"))
if not isinstance(d0_prov, Mapping):
    raise TypeError("D0 provenance must be a JSON object")
c.validate_d0_provenance_binding(d0_prov, raw_path=d0_raw, source_state=source)
c.validate_scenario_builder_manifest(
    manifest, scenarios=scenarios, input_path=input_path,
    generator_report_path=report_path, source_state=source,
    d0_raw_path=d0_raw, d0_provenance_path=d0_prov_path,
    d0_manifest_path=d0_manifest, forbidden_suite_path=frozen_suite,
    evaluation_policy_path=policy_path,
)
development_roots = c.validate_development_holdout_binding(
    holdout_path, holdout_manifest_path,
    generator_report_path=holdout_report_path, source_state=source,
    scenario_input_path=input_path, scenario_manifest_path=manifest_path,
    d0_raw_path=d0_raw, d0_provenance_path=d0_prov_path,
    d0_manifest_path=d0_manifest, forbidden_suite_path=frozen_suite,
    evaluation_policy_path=policy_path, require_model_selection_eligible=True,
)
if development_roots & (frozen_roots | d0_roots):
    raise RuntimeError("development roots overlap D0/frozen roots")
forbidden = frozen_roots | d0_roots | development_roots
c.validate_training_scenarios(scenarios, forbidden_roots=forbidden)

training_roots = []
for index, row in enumerate(scenarios):
    grouping = row.get("grouping")
    if not isinstance(grouping, Mapping):
        raise TypeError(f"scenario {index} has no grouping mapping")
    root = str(grouping.get("physical_root_fingerprint") or "").strip()
    if not root:
        raise ValueError(f"scenario {index} has no nested physical root")
    training_roots.append(root)
if len(scenarios) != 211 or len(training_roots) != len(set(training_roots)):
    raise RuntimeError("training scenario/root count contract failed")
if set(training_roots) & forbidden:
    raise RuntimeError("training roots overlap protected roots")
if len(development_roots) != 30:
    raise RuntimeError(f"expected 30 development roots, got {len(development_roots)}")

learner_seed = c.validate_training_learner_seed(model_id=str(learner), model_revision=revision)
receipt = {
    "passed": True,
    "source_commit": commit,
    "source_state": source,
    "scenario_count": len(scenarios),
    "training_physical_root_count": len(set(training_roots)),
    "development_physical_root_count": len(development_roots),
    "d0_physical_root_count": len(d0_roots),
    "frozen_physical_root_count": len(frozen_roots),
    "overlap_counts": {"training_vs_protected": len(set(training_roots) & forbidden), "development_vs_d0_frozen": len(development_roots & (frozen_roots | d0_roots))},
    "learner_seed": learner_seed,
    "bindings": {path.name: file_sha256(path) for path in (input_path, report_path, manifest_path, holdout_path, holdout_manifest_path, holdout_report_path, d0_raw, d0_prov_path, d0_manifest, frozen_suite, policy_path)},
}
payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
fd, tmp_name = tempfile.mkstemp(prefix=".input_audit.", suffix=".tmp", dir=receipt_path.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_name, receipt_path)
finally:
    if os.path.exists(tmp_name):
        os.unlink(tmp_name)
print(json.dumps(receipt, sort_keys=True))
