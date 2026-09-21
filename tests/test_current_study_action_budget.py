import hashlib
import json
from pathlib import Path

import pytest

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT
from psse_env.dagger import collect_dagger1
from psse_env.dagger.evaluation_gate import DEFAULT_POLICY_ID, load_evaluation_policy
from psse_env.dagger.study_manifest import (
    ARCHIVED_POLICY_PATH, ARCHIVED_STUDY_MANIFEST, DEFAULT_STUDY_MANIFEST,
    StudyManifestError, canonical_development_evaluation_contract,
    canonical_recovery_stress_evaluation_contract, load_study_manifest,
    validate_study_manifest,
)


def test_current_study_and_policy_share_forty_action_limit():
    current = load_study_manifest()
    assert DEFAULT_STUDY_MANIFEST.name == "dagger_multiseed_study_v2.json"
    assert DEFAULT_EPISODE_ACTION_LIMIT == 40
    for role in ("evaluation", "development_evaluation", "recovery_stress_evaluation"):
        assert current["bindings"][role]["max_steps"] == DEFAULT_EPISODE_ACTION_LIMIT
    policy = load_evaluation_policy()
    assert policy["policy_id"] == DEFAULT_POLICY_ID == "bc0_closed_loop_hard_gate_v4"
    assert policy["suite_policy"]["max_steps"] == 40
    assert policy["hard_constraints"]["maximum_steps_per_episode"] == 40
    drifted = json.loads(DEFAULT_STUDY_MANIFEST.read_text())
    drifted["bindings"]["evaluation"]["max_steps"] = 24
    with pytest.raises(StudyManifestError, match="must be 40"):
        validate_study_manifest(drifted, verify_bound_files=False)


def test_frozen_v1_retains_twenty_four_only_in_explicit_archive_context():
    original = ARCHIVED_STUDY_MANIFEST.read_bytes()
    with pytest.raises(StudyManifestError, match="archive_context=True"):
        load_study_manifest(ARCHIVED_STUDY_MANIFEST)
    historical = load_study_manifest(ARCHIVED_STUDY_MANIFEST, archive_context=True)
    for role in ("evaluation", "development_evaluation", "recovery_stress_evaluation"):
        assert historical["bindings"][role]["max_steps"] == 24
    assert historical["validation"]["archive_context"] is True
    root = Path(__file__).resolve().parents[1]
    archived_policy = root / ARCHIVED_POLICY_PATH
    assert hashlib.sha256(archived_policy.read_bytes()).hexdigest() == historical["bindings"]["evaluation"]["policy_sha256"]
    assert load_evaluation_policy(archived_policy)["suite_policy"]["max_steps"] == 24
    assert canonical_development_evaluation_contract(archive_context=True)["max_steps"] == 24
    assert canonical_recovery_stress_evaluation_contract(archive_context=True)["max_steps"] == 24
    assert ARCHIVED_STUDY_MANIFEST.read_bytes() == original


def test_production_dagger_collection_defaults_to_forty_and_rejects_old_twenty_four(monkeypatch, capsys):
    arguments = ["--input", "unused-input", "--d0-aggregate-dir", "unused-d0",
                 "--scenario-generator-report", "unused-report", "--scenario-manifest", "unused-manifest",
                 "--output", "unused-output", "--model-id", "test-model", "--model-revision", "a" * 40]
    class BudgetValidated(Exception):
        pass
    def stop_after_budget_validation(**kwargs):
        raise BudgetValidated()
    monkeypatch.setattr(collect_dagger1, "validate_collection_pass", stop_after_budget_validation)
    with pytest.raises(BudgetValidated):
        collect_dagger1.main(arguments)
    with pytest.raises(SystemExit):
        collect_dagger1.main([*arguments, "--max-steps", "24"])
    assert "requires --max-steps 40" in capsys.readouterr().err
