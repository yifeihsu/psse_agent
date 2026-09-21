from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from research.gnn_screen.dataset import content_hash, jsonable
from research.gnn_screen.wls_features import configured_case
from research.reviewed_observable_context import build_observable_context


@pytest.fixture(scope="module")
def saved_scenarios(tmp_path_factory):
    pytest.importorskip("opendssdirect")
    from pypower.api import case14
    from Transmission.generate_measurements import solve_ac_opf
    from three_phase_model.exporter import export_model
    from three_phase_model.disturbances import inject_midspan_hif
    from three_phase_model.runtime import compile_model, solve
    from three_phase_model.measurements import extract_measurements

    directory = tmp_path_factory.mktemp("reviewed_auxiliary")
    core = directory / "baseline" / "core"
    parent_id = "physical-parent"
    parent = core / "parents" / parent_id
    parent.mkdir(parents=True)
    manifest = core / "manifest.jsonl"
    manifest.write_text("")
    sigma = [0.001] * 14 + [0.01] * 108
    (core / "measurement_sigma.json").write_text(json.dumps(sigma))
    case = solve_ac_opf(case14())
    assert case is not None and case["success"]
    normal = export_model(case, parent / "model", case_id="case14")
    variants = {}
    for family in ("parameter", "topology"):
        changed = copy.deepcopy(case)
        if family == "parameter":
            changed["branch"][0, 2] *= 1.1
        else:
            changed["branch"][0, 10] = 0
        variants[family] = export_model(changed, parent / "physical_variants" / family, case_id="case14")
    rows = {}
    for family in ("healthy", "measurement", "parameter", "topology", "hif", "unbalance", "measurement+hif"):
        build = variants.get(family, normal)
        dss = compile_model(Path(build["output_dir"]) / "Master.dss")
        registry, assumptions = build["registry"], build["assumptions"]
        settings, disturbance, override = {}, None, None
        if "hif" in family:
            settings["hif"] = {"branch_row0": 2, "alpha": 0.4, "phase": 1, "resistance_pu": 10.0}
            injected = inject_midspan_hif(dss, registry, assumptions, **settings["hif"])
            override = injected["branch_overrides"]
            disturbance = {"kind": "hif", "settings": settings["hif"]}
        if family == "unbalance":
            fractions = [0.45, 0.3, 0.25]
            settings["unbalance"] = {"bus": 14, "fractions": fractions}
            for load in registry["loads"]:
                if load["bus"] == 14:
                    factor = 3 * fractions[load["phase"] - 1]
                    dss.Text.Command(f"Edit {load['element']} kW={load['kw']*factor:.16g} kvar={load['kvar']*factor:.16g}")
            solve(dss)
            disturbance = {"kind": "unbalance", **settings["unbalance"]}
        telemetry = extract_measurements(dss, registry, assumptions, branch_overrides=override)
        exact = np.array(telemetry["measurement_vector"])
        mean = exact.copy()
        audit = {"disturbance": disturbance}
        if family in variants:
            audit["actual_physical_model_path"] = Path(build["output_dir"]).relative_to(core).as_posix()
        if "measurement" in family:
            # Pure meter offsets intentionally store only multiples, to test
            # resolution of original baseline sigma through an accuracy view.
            corruption = {"channel_indices0": [16], "sigma_multiples": [12.0]}
            if family == "measurement+hif":
                corruption["additive_offsets_pu"] = [0.12]
            mean[16] += 0.12
            settings["measurement"] = corruption
            audit["measurement_corruption"] = corruption
        audit_path = parent / f"{family}_physical_audit.json"
        audit_path.write_text(json.dumps(jsonable(audit)))
        row = {"parent_id": parent_id, "window_id": f"{parent_id}:{family}",
               "case": configured_case(case), "z": mean.tolist(), "measurement_sigma": sigma,
               "measurement_kind": "noiseless_mean", "families": [] if family == "healthy" else family.split("+"),
               "offline_metadata": {"physical_audit_path": audit_path.relative_to(core).as_posix(),
                                    "settings": settings, "settings_sigma_reference_noise_profile": "baseline"}}
        rows[family] = {"row": row, "exact": exact, "telemetry": telemetry, "audit": audit_path}
    return {"directory": directory, "core": core, "manifest": manifest, "rows": rows}


@pytest.mark.parametrize("family", ["healthy", "measurement", "parameter", "topology", "hif", "unbalance", "measurement+hif"])
def test_all_families_get_identical_sensor_availability_from_actual_physics(saved_scenarios, family):
    item = saved_scenarios["rows"][family]
    row = item["row"]
    before = content_hash(row)
    metadata, receipt = build_observable_context(saved_scenarios["manifest"], row, noise_seed=1917)
    assert len(metadata["three_phase_voltages"]) == 14
    assert len(metadata["three_phase_branch_currents"]) == 20
    assert metadata["three_phase_sigma"] == 0.005
    assert metadata["branch_current_sigma_pu"] == 0.001
    assert metadata["sigma_z"] == row["measurement_sigma"]
    assert receipt["maximum_clean_scada_error"] <= 1e-8
    np.testing.assert_allclose(receipt["replayed_clean_scada"], item["exact"], rtol=0, atol=1e-8)
    assert receipt["source_unchanged"] and not receipt["scada_snapshot_modified"]
    assert content_hash(row) == before
    if family in {"parameter", "topology"}:
        assert Path(receipt["physical_model_path"]).name == family
        assert "physical_variants" in receipt["physical_model_path"]
    forbidden = ("hif", "unbalance", "j_exact", "physical_model_path", "source_window_id", "_clean", "pu_rect", "sequence_pu")
    encoded = json.dumps(metadata).lower()
    assert all(term not in encoded for term in forbidden)
    assert set(metadata) == {"three_phase_voltages", "three_phase_sigma", "three_phase_branch_currents",
                             "branch_current_sigma_pu", "sigma_z", "noise_contract"}
    assert not metadata["noise_contract"]["scada_noise_drawn_here"]
    assert "scada" not in metadata["noise_contract"]["channels"]


def test_accuracy_view_resolves_actual_variant_and_original_injection_sigma(saved_scenarios):
    view = saved_scenarios["directory"] / "accuracy_views" / "accuracy_002"
    view.mkdir(parents=True)
    manifest = view / "manifest.jsonl"
    manifest.write_text("")
    for family in ("measurement", "parameter"):
        item = saved_scenarios["rows"][family]
        row = copy.deepcopy(item["row"])
        row["measurement_sigma"] = [0.001] * 14 + [0.002] * 108
        row["offline_metadata"]["physical_audit_path"] = Path(os.path.relpath(item["audit"], view)).as_posix()
        metadata, receipt = build_observable_context(manifest, row, noise_seed=81)
        assert metadata["sigma_z"][16] == 0.002
        assert receipt["maximum_clean_scada_error"] <= 1e-8
        assert Path(receipt["original_core_path"]) == saved_scenarios["core"]
        if family == "measurement":
            assert receipt["removed_measurement_offsets"]["offsets_pu"] == pytest.approx([0.12])


def test_noise_is_reproducible_independent_of_label_and_not_clean_aliases(saved_scenarios):
    row = saved_scenarios["rows"]["healthy"]["row"]
    metadata, _ = build_observable_context(saved_scenarios["manifest"], row, noise_seed=91)
    relabeled = copy.deepcopy(row)
    relabeled["families"] = ["arbitrary_offline_label"]
    repeated, _ = build_observable_context(saved_scenarios["manifest"], relabeled, noise_seed=91)
    changed, _ = build_observable_context(saved_scenarios["manifest"], row, noise_seed=92)
    assert repeated == metadata
    assert changed["three_phase_voltages"] != metadata["three_phase_voltages"]
    assert changed["three_phase_branch_currents"] != metadata["three_phase_branch_currents"]
    # A bad SCADA meter is not an uncorrupted SCADA side-channel: additional
    # phase sensors are independent acquisitions of the same physical world.
    meter_context, _ = build_observable_context(saved_scenarios["manifest"], saved_scenarios["rows"]["measurement"]["row"], noise_seed=91)
    assert meter_context == metadata
    clean_voltage = saved_scenarios["rows"]["healthy"]["telemetry"]["three_phase_voltages"][0]
    assert metadata["three_phase_voltages"][0]["vln_pu"] != clean_voltage["vln_pu"]
    assert set(metadata["three_phase_voltages"][0]) == {"bus", "external_bus", "row0", "kvbase_ln", "vln_pu", "ang_deg"}


def test_wrong_mean_or_observed_input_fails_before_releasing_context(saved_scenarios, monkeypatch):
    import research.reviewed_observable_context as module
    row = copy.deepcopy(saved_scenarios["rows"]["healthy"]["row"])
    row["z"][0] += 0.02
    with pytest.raises(ValueError, match="differs from source mean"):
        build_observable_context(saved_scenarios["manifest"], row, noise_seed=31)
    row["measurement_kind"] = "observed"
    monkeypatch.setattr(module, "_replay", lambda *_: pytest.fail("no physical replay should occur for a noisy-reference input"))
    with pytest.raises(ValueError, match="original noiseless_mean"):
        build_observable_context(saved_scenarios["manifest"], row, noise_seed=31)
