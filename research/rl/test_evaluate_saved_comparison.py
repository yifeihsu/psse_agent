import json

import pytest

from research.rl.evaluate_saved_comparison import prepare


def saved_run(tmp_path):
    # Synthetic metadata only; no model or physical episode is evaluated here.
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"test fixture, not real weights")
    suite = tmp_path / "development.json"
    roots = [f"synthetic-{i}" for i in range(209)]
    suite.write_text(json.dumps([{"grouping": {"physical_root_fingerprint": r}} for r in roots]))
    report = tmp_path / "run.json"
    report.write_text(json.dumps({
        "paired_evaluation": {"r1_adapter": str(adapter), "paired_physical_roots": roots,
                              "seed": 20261021, "max_steps": 40},
        "research_profile": {"evidence_profile": "suspicion_gated_diagnostics",
                             "expert_variant": "ledger_ranked_compare", "hif_search_profile": "research"},
        "environment_options": {"branch_first_partial": False, "normalized_residual_threshold": 4.0},
        "model_selection": {"model_id": "fixture-only"},
        "paths": {"development_scenarios": str(suite)},
    }))
    return report, suite, tmp_path / "new-evaluation"


def test_replays_completed_adapter_and_preserves_original_inputs(tmp_path):
    report, suite, output = saved_run(tmp_path)
    before = (report.read_bytes(), suite.read_bytes())
    _, rows, manifest = prepare(report, suite, output)
    assert len(rows) == 209
    assert manifest["training"] is False
    assert manifest["seed"] == 20261021
    assert manifest["max_steps"] == 40
    assert "adapter_model.safetensors" in manifest["adapter_files"]
    assert before == (report.read_bytes(), suite.read_bytes())
    assert prepare(report, suite, output)[2] == manifest


def test_changed_weights_cannot_resume_cached_evaluation(tmp_path):
    report, suite, output = saved_run(tmp_path)
    prepare(report, suite, output)
    (tmp_path / "adapter/adapter_model.safetensors").write_bytes(b"different fixture")
    with pytest.raises(ValueError, match="another run"):
        prepare(report, suite, output)


def test_no_overwrite_of_original_results(tmp_path):
    report, suite, output = saved_run(tmp_path)
    output.mkdir()
    (output / "existing.json").write_text("{}")
    with pytest.raises(ValueError, match="another run"):
        prepare(report, suite, output)


def test_original_root_order_required(tmp_path):
    report, suite, output = saved_run(tmp_path)
    suite.write_text(json.dumps(list(reversed(json.loads(suite.read_text())))))
    with pytest.raises(ValueError, match="original 209 roots"):
        prepare(report, suite, output)
