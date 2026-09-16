"""Physical corpus contracts; these are not detector accuracy claims."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opendssdirect")

from psse_env.systems import resolve_system
from research.gnn_screen.dataset import load_manifest
from research.gnn_screen.generate_corpus import (
    connected, competing_variant, generate_corpus, healthy_audit,
    physical_snapshot, sample_parent,
)
from three_phase_model.exporter import export_model


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("physical_corpus") / "run"
    report = generate_corpus(root, parents_by_split={s: 1 for s in
        ("train", "validation", "calibration", "test")}, seed=20260916,
        noise_replicates=2, healthy_calibration_replicates=7,
        healthy_replicates_by_split={"validation": 3, "test": 4})
    return root, report, load_manifest(root / "manifest.jsonl")


def test_physical_pipeline_grouping_and_same_noise_source(generated):
    root, report, rows = generated
    assert not report["failed_variants"]
    assert len(rows) == 55
    assert {tuple(r["families"]) for r in rows} >= {
        (), ("hif",), ("unbalance",), ("measurement",), ("parameter",), ("topology",),
        ("hif", "measurement"), ("unbalance", "parameter"), ("hif", "unbalance"), ("hif", "topology")}
    assert len({r["parent_id"] for r in rows}) == 4
    for parent in {r["parent_id"] for r in rows}:
        group = [r for r in rows if r["parent_id"] == parent]
        assert len({r["split"] for r in group}) == 1
        assert len({r["offline_metadata"]["physical_case_hash"] for r in group}) == 1
    assert len({tuple(r["measurement_sigma"]) for r in rows}) == 1
    assert len({r["offline_metadata"]["measurement_source"] for r in rows}) == 1
    for row in rows:
        assert len(row["z"]) == 122
        assert len(row["case"]["bus"]) == 14
        assert len(row["case"]["branch"]) == 20
        assert row["measurement_kind"] == "noiseless_mean"
        assert row["offline_metadata"]["affected_phase"]
        opf = row["offline_metadata"]["operating_point"]["source_opf"]
        assert opf["success"] and opf["reference_q_limits_enforced"]
        assert max(opf["maximum_bound_violations"].values()) < .01
        audit = json.loads((root / row["offline_metadata"]["physical_audit_path"]).read_text())
        assert audit["external_kcl_max_mismatch_pu"] < 1e-7
        if row["split"] == "calibration":
            assert row["families"] == [] and row["noise_replicates"] == 7
        elif row["split"] == "test" and not row["families"]:
            assert row["noise_replicates"] == 4
    assert report["healthy_max_balanced_equation_error_pu"] < 1e-8


def test_disabled_hif_split_controls_and_unbalance_conserve_power(generated):
    root, _, rows = generated
    controls = list(root.glob("parents/*/disabled_hif_split_control_audit.json"))
    assert len(controls) == 3
    for path in controls:
        audit = json.loads(path.read_text())
        assert audit["maximum_unsplit_measurement_difference_pu"] < 1e-8
        assert audit["physics"]["disturbances"][0]["audit"]["passed"]
        assert not audit["physics"]["disturbances"][0]["receipt"]["fault_enabled"]
    for row in rows:
        if "unbalance" not in row["families"]:
            continue
        physics = json.loads((root / row["offline_metadata"]["physical_audit_path"]).read_text())
        disturbance = next(d for d in physics["disturbances"] if d["kind"] == "phase_load_redistribution")
        assert sum(disturbance["phase_factors"].values()) == pytest.approx(3)
        assert row["offline_metadata"]["unbalance_increased_phase"] in (1, 2, 3)


@pytest.mark.parametrize("branch_row", [0, 7])
def test_disabled_line_or_transformer_keeps_zero_fixed_terminal_channels(tmp_path, branch_row):
    case = resolve_system("case14").load_case()
    case["branch"][branch_row, 10] = 0
    assert connected(case)
    build = export_model(case, tmp_path / "model", case_id="case14")
    z, _ = physical_snapshot(build)
    assert healthy_audit(build, z)["passed"]
    assert z[[42 + branch_row, 62 + branch_row, 82 + branch_row, 102 + branch_row]] == pytest.approx([0] * 4)


def test_competing_model_errors_preserve_physical_measurement_identity():
    rng = np.random.default_rng(9)
    case, _ = sample_parent(rng, 0)
    z = np.linspace(.1, 1.2, 122)
    for family in ("parameter", "topology"):
        candidate, observed, metadata = competing_variant(case, z, family=family,
            severity="intermediate", rng=rng)
        np.testing.assert_array_equal(observed, z)
        np.testing.assert_array_equal(candidate["branch"][:, :2], case["branch"][:, :2])
        np.testing.assert_array_equal(candidate["bus"][:, 0], case["bus"][:, 0])
        assert metadata["physical_model_unchanged"]
        assert connected(candidate)
    candidate, observed, _ = competing_variant(case, z, family="measurement",
        severity="intermediate", rng=rng)
    np.testing.assert_array_equal(candidate["branch"], case["branch"])
    assert np.count_nonzero(observed != z) == 1


def test_generation_refuses_overwrite_and_invalid_healthy_replicates(tmp_path):
    out = tmp_path / "existing"
    out.mkdir()
    marker = out / "important.txt"
    marker.write_text("preserve")
    with pytest.raises(FileExistsError):
        generate_corpus(out, parents_by_split={"train": 1})
    assert marker.read_text() == "preserve"
    with pytest.raises(ValueError, match="healthy replicate"):
        generate_corpus(tmp_path / "invalid", parents_by_split={"train": 1},
            healthy_replicates_by_split={"test": 0})
