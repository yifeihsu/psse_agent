"""DAgger-aligned screen corpus: source fidelity, DAgger error rules, grouping and determinism."""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

pytest.importorskip("opendssdirect")
from research.gnn_screen import dagger_corpus as dc
from research.gnn_screen.dataset import load_manifest

pytestmark = pytest.mark.skipif(
    not all((dc.ARTIFACTS / name / "samples.jsonl").is_file()
            for name in (dc.DEFAULT_HIF_CORPORA[0], dc.DEFAULT_UNBALANCE_CORPUS)),
    reason="DAgger physical corpora are not present")


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    path = tmp_path_factory.mktemp("dagger_aligned") / "corpus"
    summary = dc.build_corpus(path, max_parents_per_corpus=2, evaluation_hif=dc.DEFAULT_EVALUATION_HIF[:1])
    raw = [json.loads(line) for line in (path / "manifest.jsonl").open(encoding="utf-8")]
    evaluation = [json.loads(line) for line in (path / "hif_resistance_evaluation_manifest.jsonl").open(encoding="utf-8")]
    return path, summary, raw, evaluation


def _by_window(rows):
    return {row["window_id"]: row for row in rows}


def test_hif_and_unbalance_means_are_the_stored_dagger_vectors(corpus):
    _, _, rows, _ = corpus
    windows = _by_window(rows)
    for name in dc.DEFAULT_HIF_CORPORA:
        for source in dc.fault_rows(name)[:2]:
            parent = f"dagger:{name}:{source['id']}"
            if f"{parent}:s0:healthy" not in windows:
                continue  # calibration parents keep healthy windows only
            for scan in source["scans"]:
                k = scan["scan_index"]
                if f"{parent}:s{k}:hif" in windows:
                    assert windows[f"{parent}:s{k}:hif"]["z"] == scan["z_clean"]
            assert windows[f"{parent}:s0:healthy"]["z"] == source["z_true"]
    for source in dc.fault_rows(dc.DEFAULT_UNBALANCE_CORPUS)[:2]:
        parent = f"dagger:{dc.DEFAULT_UNBALANCE_CORPUS}:{source['id']}"
        assert windows[f"{parent}:healthy"]["z"] == source["z_true"]
        if f"{parent}:unbalance" in windows:
            assert windows[f"{parent}:unbalance"]["z"] == source["z_clean"]


def test_corpus_from_different_physics_is_refused():
    legacy = "hif_physical69_main_train_84x10_20260919"
    if not (dc.ARTIFACTS / legacy / "samples.jsonl").is_file():
        pytest.skip("pre-fix HIF corpus not present")
    row = dc.fault_rows(legacy)[0]
    task = {"kind": "hif", "row": row, "corpus": legacy, "detectable": False, "seed": 1,
            "sigma": dc.measurement_sigma().tolist(), "case": dc.configured_case14()}
    with pytest.raises(ValueError, match="different physics"):
        dc.build_hif_parent(task)


def test_resimulated_healthy_scans_use_the_generators_own_paths():
    hif = dc.fault_rows(dc.DEFAULT_HIF_CORPORA[0])[0]
    unbalance = dc.fault_rows(dc.DEFAULT_UNBALANCE_CORPUS)[0]
    assert dc.simulator("hif_operating_point").solve(hif["scans"][0]["op_point"]) == hif["z_true"]
    op = dc.unbalance_operating_point(unbalance["op_point"])
    # The 2026-09-23opf rows store the OPF dispatch; the replay applies it after the load scaling.
    assert {"load_scale", "generator_dispatch_kw", "voltage_setpoints_pu", "source_voltage_pu"} == set(op)
    assert "target_bus" not in op
    assert dc.simulator("imbalance_balanced").solve(op) == unbalance["z_true"]
    # Without the stored dispatch the checked-in model dispatch would be replayed instead.
    assert dc.simulator("imbalance_balanced").solve({"load_scale": op["load_scale"]}) != unbalance["z_true"]


def test_parent_grouping_splits_and_labels(corpus):
    path, summary, rows, _ = corpus
    loaded = load_manifest(path / "manifest.jsonl")
    assert len(loaded) == len(rows) and summary["simulation_failures"] == 0
    splits = {}
    for row in rows:
        assert splits.setdefault(row["parent_id"], row["split"]) == row["split"]
        meta = row["offline_metadata"]
        if row["split"] == "calibration":
            assert row["families"] == []
        if row["families"] == []:
            assert row["severity"] == "healthy" and meta["noiseless_wls_J"] < 1e-3
        elif row["split"] == "train":
            assert row["severity"] == dc.TRAIN_SEVERITY
        else:
            assert row["severity"] == meta["stratum"]
        assert meta["generator_reactive_limits_reset"] is False
    hif_parents = [p for p in splits if "hif_physical" in p and splits[p] != "calibration"]
    for parent in hif_parents:
        group = [r for r in rows if r["parent_id"] == parent]
        combos = Counter("+".join(r["families"]) or "healthy" for r in group)
        assert combos["healthy"] == 10 and combos["hif"] == 10
        assert combos["hif+measurement"] == 2 and combos["measurement"] == 2
        assert combos["parameter"] == 2 and combos["topology"] == 2
        assert combos["measurement+parameter"] == 1 and combos["measurement+topology"] == 1


def test_mixed_meter_errors_skip_the_faulted_branch_flows(corpus):
    _, _, rows, _ = corpus
    mixed = [r for r in rows if r["families"] in (["measurement", "parameter"], ["measurement", "topology"])]
    assert mixed
    for row in mixed:
        meta = row["offline_metadata"]
        fault = meta.get("parameter") or meta.get("topology")
        assert not set(meta["measurement"]["channel_indices0"]) & dc.branch_flow_indices(fault["branch_row0"])
        assert meta["measurement"]["meter_count"] == 1


def test_evaluation_manifest_keeps_hif_scans_only(corpus):
    _, _, _, evaluation = corpus
    assert {r["split"] for r in evaluation} == {"test"}
    assert {tuple(r["families"]) for r in evaluation} == {(), ("hif",)}
    assert all(r["offline_metadata"]["evaluation_only"] for r in evaluation)


def test_dagger_parameter_rule_and_meter_distribution():
    case = dc.configured_case14()
    assert dc.parameter_rows(case) == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 15, 16, 17, 18, 19]
    assert 13 not in dc.topology_rows()
    rng = np.random.default_rng(3)
    for _ in range(300):
        spec = dc.sample_parameter(rng, dc.parameter_rows(case))
        for component, factor in (("R", spec["r_factor"]), ("X", spec["x_factor"])):
            if component in spec["subtype"]:
                assert 0.1 <= factor <= 0.5 or 2.0 <= factor <= 5.0
            else:
                assert factor == 1.0
    sigma = dc.measurement_sigma()
    z = np.zeros(122).tolist()
    for count in (1, None):
        for _ in range(100):
            blocked = dc.branch_flow_indices(4)
            changed, info = dc.meter_error(z, rng, sigma, count=count, blocked=blocked)
            multiples = np.asarray(changed)[info["channel_indices0"]] / sigma[info["channel_indices0"]]
            assert np.all((np.abs(multiples) >= 10) & (np.abs(multiples) <= 15))
            assert np.count_nonzero(changed) == info["meter_count"] and not set(info["channel_indices0"]) & blocked
            assert (info["meter_count"] == 1) if count == 1 else (2 <= info["meter_count"] <= 5)


def test_strata_and_deterministic_split():
    assert dc.hif_stratum({"resistance_ohm": 150.0}) == "hif_69kv_100_200_ohm"
    assert dc.hif_stratum({"resistance_ohm": 1000.0}) == "hif_69kv_500_1000_ohm"
    assert dc.hif_stratum({"resistance_ohm": 60.0, "local_kv_ll": 13.8}) == "hif_13p8kv_0_100_ohm"
    parents = [{"parent_id": f"p{i}", "family": "hif", "stratum": "s", "detectable": False} for i in range(200)]
    first, second = dc.assign_splits(parents, 7), dc.assign_splits(list(reversed(parents)), 7)
    assert first == second
    shares = Counter(first.values())
    for name, fraction in dc.SPLIT_FRACTIONS.items():
        assert abs(shares[name] - 200 * fraction) <= 1
